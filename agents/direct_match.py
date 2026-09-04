"""
Direct Match Agent -- takes Cluster Agent's proposed clusters and
confirms/rejects them via LLM reasoning. The ONLY agent with reject-and-loop
power (max 2 retries) -- on rejection it calls cluster.reproposal() with the
rejection reason and tries again.

Select-based matching (per "Match, Compare, or Select?", COLING 2025): the
LLM is given the ledger record plus candidates and asked to pick the correct
one, not asked an open-ended yes/no per pair. EVERY cluster goes through
this LLM step -- no rule-based shortcut skips the LLM for "obvious" matches
(deliberate decision against agent-specific cost tiering); cost control
comes from batching multiple clusters per API call instead.

Every LLM "confirm" is independently re-verified in code before being
trusted (_verify_confirm_decision) -- caught the model hallucinating a false
amount match during testing, twice. Never trust an LLM's numeric claim
without checking it.
"""

import hashlib
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for sibling agents/ imports

import config
import cluster
from env_setup import get_groq_client

GROQ_MODEL = "openai/gpt-oss-20b"
BATCH_SIZE = 8
MAX_RETRIES = 2
MAX_API_ATTEMPTS = 3
_CACHE_BASE = os.environ.get("RECON_CACHE_ROOT", os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_CACHE_BASE, "llm_cache")


def _cache_key(c):
    payload = json.dumps({
        "member_record_ids": sorted(c["member_record_ids"]),
        "retry_count": c["retry_count"],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _cache_path(c):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{_cache_key(c)}.json")


def _load_cached(c):
    path = _cache_path(c)
    if os.path.exists(path):
        with open(path) as f:
            decision = json.load(f)
        if decision.get("reason") == "stub":
            print(f"WARNING: discarding stub-tagged cache entry at {path} -- forcing live call.")
            return None
        return decision
    return None


def _save_cache(c, decision):
    with open(_cache_path(c), "w") as f:
        json.dump(decision, f)


def _record_summary(rec, source):
    if source == "ledger":
        return {"order_id": rec["order_id"], "amount_paise": rec["amount"],
                "booked_date": rec["booked_date"].isoformat(), "counterparty": rec["counterparty"],
                "narration": rec["narration"]}
    if source == "gateway":
        return {"order_id": rec["order_id"], "amount_paise": rec["amount"],
                "created_at": rec["created_at"].isoformat(), "method": rec["method"], "notes": rec["notes"]}
    if source == "settlement":
        return {"amount_paise": rec["amount"], "created_at": rec["created_at"].isoformat(),
                "narration": rec["narration"]}


def _build_cluster_payload(c, ledger_by_id, gateway_by_id, settlement_by_id):
    ledger_id = c["member_record_ids"][0]
    return {
        "cluster_id": c["cluster_id"],
        "ledger_record": _record_summary(ledger_by_id[ledger_id], "ledger"),
        "gateway_candidates": [
            {"record_id": cand["record_id"], "match_method": cand["match_method"],
             **_record_summary(gateway_by_id[cand["record_id"]], "gateway")}
            for cand in c["gateway_candidates"]
        ],
        "settlement_candidates": [
            {"record_id": cand["record_id"], "match_method": cand["match_method"],
             **_record_summary(settlement_by_id[cand["record_id"]], "settlement")}
            for cand in c["settlement_candidates"]
        ],
    }


def _build_prompt(cluster_payloads):
    tolerance_paise = config.CONFIRM_AMOUNT_TOLERANCE_PAISE
    tolerance_pct = config.CONFIRM_AMOUNT_TOLERANCE_PERCENT
    instructions = f"""You are reconciling payment records for a finance system. For each cluster below,
you are given ONE internal ledger record and its CANDIDATE gateway/settlement records
(found by a prior narrowing step -- they are plausible, not guaranteed correct).

For each cluster, decide:
1. Which gateway candidate (if any) is the TRUE match for the ledger record, or "none".
2. Which settlement candidate (if any) is the TRUE match, or "none".
3. An overall action: "confirm" (the picks are correct matches) or "reject" (nothing
   fits well enough, or the ledger record looks unmatched).

Matching rule: amounts match if they are within Rs{tolerance_paise/100:.2f} OR
{tolerance_pct*100:.1f}% of each other, whichever is SMALLER. Dates should be close
and consistent with the transaction narration/notes.

Each candidate has a "match_method" field:
- "amount_date_fallback" means it's a guess based on amount+date proximity only --
  verify it carefully using amount/date/narration evidence; it is NOT guaranteed.
- "order_id" (gateway candidates) means the reference/order number is confirmed
  correct -- but this does NOT mean the amount can be ignored. If there is exactly
  ONE gateway candidate for this order_id and its amount matches the ledger amount
  within tolerance, that is a strong confirm. If there are MULTIPLE gateway
  candidates sharing the same order_id, check whether any SINGLE candidate's
  amount matches the full ledger amount within tolerance -- if none do, this is a
  split payment (the ledger amount was paid across multiple legs) and the correct
  action is "reject", since resolving split payments is handled by a separate,
  later step. Do NOT pick one leg and treat it as if it were the full match.
- "settlement_id" (settlement candidates) means this payment was reliably confirmed
  as part of that settlement BATCH -- a batch aggregates MANY payments swept
  together, so its total amount will almost always be LARGER than this single
  ledger amount. That size difference is normal and expected, NOT a mismatch. Do
  not reject, and do not require the settlement amount to equal the ledger amount
  -- its presence alone is sufficient corroboration. The GATEWAY side amount match
  against the ledger is what actually decides confirm/reject; the settlement
  candidate is supporting evidence, not the deciding number.

Respond with ONLY a JSON object (no markdown fences, no prose) with a single key
"decisions" containing an array, one object per cluster:
{{
  "decisions": [
    {{
      "cluster_id": "...",
      "action": "confirm" | "reject",
      "chosen_gateway_record_id": "..." | null,
      "chosen_settlement_record_id": "..." | null,
      "reason": "short, specific reason referencing the actual numbers/dates/narration"
    }}
  ]
}}

Clusters:
{json.dumps(cluster_payloads, indent=2)}
"""
    return instructions


def _call_groq(prompt):
    client = get_groq_client()
    last_error = None
    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"},
                reasoning_effort="low",
                reasoning_format="hidden",
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt < MAX_API_ATTEMPTS:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Groq API failed after {MAX_API_ATTEMPTS} attempts: {last_error}")


def _parse_response(raw_text):
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    parsed = json.loads(text.strip())
    if isinstance(parsed, dict) and "decisions" in parsed:
        return parsed["decisions"]
    return parsed


def _within_confirm_tolerance(a, b):
    diff = abs(a - b)
    pct_tolerance = min(a, b) * config.CONFIRM_AMOUNT_TOLERANCE_PERCENT
    tolerance = min(config.CONFIRM_AMOUNT_TOLERANCE_PAISE, pct_tolerance)
    return diff <= tolerance


def _verify_confirm_decision(decision, c, ledger_by_id, gateway_by_id):
    """Deterministic backstop -- never trust the LLM's "confirm" on faith."""
    chosen_gw_id = decision.get("chosen_gateway_record_id")
    if not chosen_gw_id:
        return True, None

    ledger_id = c["member_record_ids"][0]
    ledger_amount = ledger_by_id[ledger_id]["amount"]
    gw_amount = gateway_by_id[chosen_gw_id]["amount"]

    if _within_confirm_tolerance(gw_amount, ledger_amount):
        return True, None

    return False, (
        f"system override: LLM confirmed with chosen_gateway_record_id={chosen_gw_id} "
        f"(amount {gw_amount}) but this does not satisfy confirm tolerance against "
        f"ledger amount {ledger_amount} -- treating as reject"
    )


def call_llm_batch(clusters_batch, ledger_by_id, gateway_by_id, settlement_by_id):
    if not clusters_batch:
        return {}
    payloads = [
        _build_cluster_payload(c, ledger_by_id, gateway_by_id, settlement_by_id)
        for c in clusters_batch
    ]
    prompt = _build_prompt(payloads)
    raw = _call_groq(prompt)
    decisions = _parse_response(raw)
    return {d["cluster_id"]: d for d in decisions}


def _chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def run_direct_match(clusters, dataset, decision_log, llm_batch_fn=call_llm_batch):
    ledger_by_id = {r["record_id"]: r for r in dataset["ledger"]}
    gateway_by_id = {r["record_id"]: r for r in dataset["gateway"]}
    settlement_by_id = {r["record_id"]: r for r in dataset["settlement"]}

    def log(cluster_id, decision, reason):
        decision_log.setdefault(cluster_id, []).append({
            "agent": "direct_match", "decision": decision, "reason": reason,
        })

    pending_ids = [cid for cid, c in clusters.items() if c["status"] == "proposed"]
    round_num = 0

    while pending_ids and round_num <= MAX_RETRIES:
        round_num += 1
        next_round_ids = []

        for batch_ids in _chunks(pending_ids, BATCH_SIZE):
            batch = [clusters[cid] for cid in batch_ids]
            decisions = {}
            uncached = []
            for c in batch:
                cached = _load_cached(c)
                if cached:
                    decisions[c["cluster_id"]] = cached
                else:
                    uncached.append(c)

            if uncached:
                try:
                    fresh = llm_batch_fn(uncached, ledger_by_id, gateway_by_id, settlement_by_id)
                    for c in uncached:
                        d = fresh.get(c["cluster_id"])
                        if d:
                            _save_cache(c, d)
                            decisions[c["cluster_id"]] = d
                except Exception as e:
                    for c in uncached:
                        c["status"] = "unresolved_system_error"
                        log(c["cluster_id"], "unresolved_system_error", f"API failure: {e}")
                    continue

            for cid, decision in decisions.items():
                c = clusters[cid]
                action = decision.get("action")
                reason = decision.get("reason", "")

                if action == "confirm":
                    is_valid, override_reason = _verify_confirm_decision(
                        decision, c, ledger_by_id, gateway_by_id
                    )
                    if not is_valid:
                        action = "reject"
                        reason = override_reason

                if action == "confirm":
                    c["status"] = "confirmed"
                    c["chosen_gateway_record_id"] = decision.get("chosen_gateway_record_id")
                    c["chosen_settlement_record_id"] = decision.get("chosen_settlement_record_id")
                    log(cid, "confirmed", reason)
                else:
                    log(cid, "rejected", reason)
                    if c["retry_count"] >= MAX_RETRIES:
                        c["status"] = "unresolved_direct_match_exhausted"
                    else:
                        cluster.reproposal(c, reason)
                        next_round_ids.append(cid)

        pending_ids = next_round_ids

    for cid in pending_ids:
        clusters[cid]["status"] = "unresolved_direct_match_exhausted"

    return clusters
