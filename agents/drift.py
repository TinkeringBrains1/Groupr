"""
Drift Agent -- FINAL pipeline stage. Two jobs:
1. Hard impossibility check (deterministic): a gateway/settlement date
   before the ledger's booked_date is flagged straight to
   "flagged_impossible" -- no LLM call, no retry.
2. LLM final judgment on whatever passes: "resolve" (legitimate, even if
   incomplete/unusual) or "exception" (needs human review). Terminal -- no
   retry loop.

Every "resolve" is independently re-verified in code before being trusted
(_verify_resolve_decision) -- Drift is the last stage with no downstream
catch, so an unverified hallucination here has no safety net at all.
"""

import hashlib
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from env_setup import get_groq_client

GROQ_MODEL = "openai/gpt-oss-20b"
MAX_API_ATTEMPTS = 3
_CACHE_BASE = os.environ.get("RECON_CACHE_ROOT", os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_CACHE_BASE, "llm_cache_drift")


def check_impossibility(cluster, ledger_by_id, gateway_by_id, settlement_by_id):
    ledger_id = cluster["member_record_ids"][0]
    ledger_date = ledger_by_id[ledger_id]["booked_date"]

    for c in cluster.get("gateway_candidates", []):
        gw = gateway_by_id.get(c["record_id"])
        if gw and gw["created_at"] < ledger_date:
            return (
                f"impossible: gateway payment {c['record_id']} dated {gw['created_at'].isoformat()} "
                f"is BEFORE the ledger booking date {ledger_date.isoformat()}"
            )

    for c in cluster.get("settlement_candidates", []):
        st = settlement_by_id.get(c["record_id"])
        if st and st["created_at"] < ledger_date:
            return (
                f"impossible: settlement {c['record_id']} dated {st['created_at'].isoformat()} "
                f"is BEFORE the ledger booking date {ledger_date.isoformat()}"
            )

    return None


def _record_summary(rec, source):
    if source == "ledger":
        return {"order_id": rec["order_id"], "amount_paise": rec["amount"],
                "booked_date": rec["booked_date"].isoformat(), "counterparty": rec["counterparty"],
                "narration": rec["narration"]}
    if source == "gateway":
        return {"order_id": rec["order_id"], "amount_paise": rec["amount"],
                "created_at": rec["created_at"].isoformat(), "method": rec["method"], "notes": rec["notes"]}
    return {"amount_paise": rec["amount"], "created_at": rec["created_at"].isoformat(),
            "narration": rec["narration"]}


def _build_prompt(cluster, ledger_by_id, gateway_by_id, settlement_by_id, history):
    ledger_id = cluster["member_record_ids"][0]
    payload = {
        "ledger_record": _record_summary(ledger_by_id[ledger_id], "ledger"),
        "gateway_candidates": [
            {"record_id": c["record_id"], "match_method": c["match_method"],
             **_record_summary(gateway_by_id[c["record_id"]], "gateway")}
            for c in cluster.get("gateway_candidates", []) if c["record_id"] in gateway_by_id
        ],
        "settlement_candidates": [
            {"record_id": c["record_id"], "match_method": c["match_method"],
             **_record_summary(settlement_by_id[c["record_id"]], "settlement")}
            for c in cluster.get("settlement_candidates", []) if c["record_id"] in settlement_by_id
        ],
        "prior_agent_history": history,
    }
    return f"""This is the FINAL stage of a payment reconciliation pipeline. Earlier
stages already tried and could not confidently resolve this ledger record --
their reasoning is included below as prior_agent_history. This is the last
chance to resolve it; there is no further retry after this.

Make ONE final call:
- "resolve": treat this as a legitimate match, even if incomplete or unusual.
- "exception": this genuinely needs human review.

Be honest, not optimistic: a wrong "resolve" is a worse outcome than an
honest "exception".

Data:
{json.dumps(payload, indent=2)}

Respond with ONLY a JSON object, no markdown fences, no prose:
{{
  "action": "resolve" | "exception",
  "chosen_gateway_record_ids": ["..."] | [],
  "chosen_settlement_record_id": "..." | null,
  "reason": "short, specific reason referencing the actual evidence"
}}

If "resolve", chosen_gateway_record_ids/chosen_settlement_record_id MUST name the
specific candidate(s) from above that support your decision -- do not resolve
without naming which evidence you used. If "exception", these can be empty/null.
"""


def _call_groq(prompt):
    client = get_groq_client()
    last_error = None
    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
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
    return json.loads(text.strip())


def _within_confirm_tolerance(a, b):
    diff = abs(a - b)
    pct_tolerance = min(a, b) * config.CONFIRM_AMOUNT_TOLERANCE_PERCENT
    tolerance = min(config.CONFIRM_AMOUNT_TOLERANCE_PAISE, pct_tolerance)
    return diff <= tolerance


def _verify_resolve_decision(decision, cluster, ledger_by_id, gateway_by_id, settlement_by_id):
    """Same principle as Direct Match's verification -- Drift is terminal
    with no downstream catch, so an unverified "resolve" has no safety net."""
    ledger_id = cluster["member_record_ids"][0]
    ledger_amount = ledger_by_id[ledger_id]["amount"]

    gw_ids = decision.get("chosen_gateway_record_ids") or []
    settle_id = decision.get("chosen_settlement_record_id")

    if not gw_ids and not settle_id:
        return False, "system override: Drift said 'resolve' but named no evidence -- treating as exception"

    if gw_ids:
        valid_ids = {c["record_id"] for c in cluster.get("gateway_candidates", [])}
        unknown = [g for g in gw_ids if g not in valid_ids or g not in gateway_by_id]
        if unknown:
            return False, f"system override: chosen_gateway_record_ids references unknown record(s) {unknown}"
        leg_sum = sum(gateway_by_id[g]["amount"] for g in gw_ids)
        if not _within_confirm_tolerance(leg_sum, ledger_amount):
            return False, (
                f"system override: Drift claimed gateway legs {gw_ids} (sum {leg_sum}) resolve the "
                f"ledger amount {ledger_amount}, but this does not satisfy confirm tolerance"
            )
        return True, None

    valid_settle_ids = {c["record_id"] for c in cluster.get("settlement_candidates", [])}
    if settle_id not in valid_settle_ids or settle_id not in settlement_by_id:
        return False, f"system override: chosen_settlement_record_id {settle_id} is not a known candidate"
    settle_amount = settlement_by_id[settle_id]["amount"]
    if not _within_confirm_tolerance(settle_amount, ledger_amount):
        return False, (
            f"system override: Drift claimed settlement {settle_id} (amount {settle_amount}) matches "
            f"ledger amount {ledger_amount}, but this does not satisfy confirm tolerance"
        )
    return True, None


def _cache_key(cluster_id, history_len):
    payload = json.dumps({"cluster_id": cluster_id, "history_len": history_len}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _cache_path(cluster_id, history_len):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{_cache_key(cluster_id, history_len)}.json")


def _load_cached(cluster_id, history_len):
    path = _cache_path(cluster_id, history_len)
    if os.path.exists(path):
        with open(path) as f:
            decision = json.load(f)
        if decision.get("reason") == "stub":
            print(f"WARNING: discarding stub-tagged cache entry at {path} -- forcing live call.")
            return None
        return decision
    return None


def _save_cache(cluster_id, history_len, decision):
    with open(_cache_path(cluster_id, history_len), "w") as f:
        json.dump(decision, f)


def run_drift(clusters, dataset, decision_log, llm_call_fn=None):
    ledger_by_id = {r["record_id"]: r for r in dataset["ledger"]}
    gateway_by_id = {r["record_id"]: r for r in dataset["gateway"]}
    settlement_by_id = {r["record_id"]: r for r in dataset["settlement"]}

    def log(cluster_id, decision, reason):
        decision_log.setdefault(cluster_id, []).append({
            "agent": "drift", "decision": decision, "reason": reason,
        })

    candidates = [
        (cid, c) for cid, c in clusters.items()
        if c["status"] == "unresolved_direct_match_exhausted"
    ]

    for cid, cluster in candidates:
        impossibility = check_impossibility(cluster, ledger_by_id, gateway_by_id, settlement_by_id)
        if impossibility:
            cluster["status"] = "flagged_impossible"
            log(cid, "flagged_impossible", impossibility)
            continue

        history = decision_log.get(cid, [])
        history_len = len(history)
        cached = _load_cached(cid, history_len)
        try:
            if cached:
                decision = cached
            else:
                prompt = _build_prompt(cluster, ledger_by_id, gateway_by_id, settlement_by_id, history)
                call = llm_call_fn or _call_groq
                raw = call(prompt)
                decision = _parse_response(raw)
                _save_cache(cid, history_len, decision)
        except Exception as e:
            cluster["status"] = "unresolved_system_error"
            log(cid, "unresolved_system_error", f"API failure: {e}")
            continue

        if decision.get("action") == "resolve":
            is_valid, override_reason = _verify_resolve_decision(
                decision, cluster, ledger_by_id, gateway_by_id, settlement_by_id
            )
            if is_valid:
                cluster["status"] = "confirmed"
                cluster["drift_resolution"] = "resolved_by_drift"
                cluster["chosen_gateway_record_ids"] = decision.get("chosen_gateway_record_ids") or []
                cluster["chosen_settlement_record_id"] = decision.get("chosen_settlement_record_id")
                log(cid, "resolved", decision.get("reason", ""))
            else:
                cluster["status"] = "final_exception"
                log(cid, "final_exception", override_reason)
        else:
            cluster["status"] = "final_exception"
            log(cid, "final_exception", decision.get("reason", ""))

    return clusters
