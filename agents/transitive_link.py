"""
Transitive Link Agent -- picks up clusters Direct Match couldn't resolve,
specifically split payments: does any SUBSET of 2-3 gateway legs sum to the
ledger amount? Candidate pool is Blocking's own gateway_candidates (already
order_id-matched) -- no fresh counterparty+date search (schema doesn't
support it: gateway has no counterparty field).

Date window: 10 days (deliberately narrower than the generator's worst-case
~20-day staged-split spread -- legs spread wider are left for Drift).

No reject power -- unconfirmed clusters pass through untouched to Drift.
"""

import hashlib
import itertools
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
DATE_WINDOW_DAYS = 10
MAX_SUBSET_SIZE = 3
_CACHE_BASE = os.environ.get("RECON_CACHE_ROOT", os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_CACHE_BASE, "llm_cache_transitive")


def _within_tolerance(a, b):
    diff = abs(a - b)
    pct_tolerance = min(a, b) * config.CONFIRM_AMOUNT_TOLERANCE_PERCENT
    tolerance = min(config.CONFIRM_AMOUNT_TOLERANCE_PAISE, pct_tolerance)
    return diff <= tolerance


def _within_date_window(legs, gateway_by_id):
    dates = [gateway_by_id[leg["record_id"]]["created_at"] for leg in legs]
    return (max(dates) - min(dates)).days <= DATE_WINDOW_DAYS


def find_subset_sum_candidates(cluster, ledger_by_id, gateway_by_id):
    ledger_id = cluster["member_record_ids"][0]
    ledger_amount = ledger_by_id[ledger_id]["amount"]
    legs = cluster["gateway_candidates"]

    if len(legs) < 2:
        return []

    valid = []
    for size in range(2, min(MAX_SUBSET_SIZE, len(legs)) + 1):
        for combo in itertools.combinations(legs, size):
            total = sum(gateway_by_id[leg["record_id"]]["amount"] for leg in combo)
            if _within_tolerance(total, ledger_amount) and _within_date_window(combo, gateway_by_id):
                dates = [gateway_by_id[leg["record_id"]]["created_at"] for leg in combo]
                spread = (max(dates) - min(dates)).days
                valid.append((spread, [leg["record_id"] for leg in combo]))

    valid.sort(key=lambda x: x[0])
    return [v[1] for v in valid]


def _cache_key(cluster_id, leg_ids):
    payload = json.dumps({"cluster_id": cluster_id, "legs": sorted(leg_ids)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _cache_path(cluster_id, leg_ids):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{_cache_key(cluster_id, leg_ids)}.json")


def _load_cached(cluster_id, leg_ids):
    path = _cache_path(cluster_id, leg_ids)
    if os.path.exists(path):
        with open(path) as f:
            decision = json.load(f)
        if decision.get("reason") == "stub":
            print(f"WARNING: discarding stub-tagged cache entry at {path} -- forcing live call.")
            return None
        return decision
    return None


def _save_cache(cluster_id, leg_ids, decision):
    with open(_cache_path(cluster_id, leg_ids), "w") as f:
        json.dump(decision, f)


def _record_summary(rec, source):
    if source == "ledger":
        return {"order_id": rec["order_id"], "amount_paise": rec["amount"],
                "booked_date": rec["booked_date"].isoformat(), "narration": rec["narration"]}
    return {"order_id": rec["order_id"], "amount_paise": rec["amount"],
            "created_at": rec["created_at"].isoformat(), "method": rec["method"], "notes": rec["notes"]}


def _build_prompt(ledger_summary, leg_summaries, total):
    return f"""A ledger record was NOT matched to a single gateway payment. We found a
COMBINATION of gateway payments whose amounts sum exactly to the ledger amount
(this arithmetic is already verified correct) -- your job is only to judge
whether the evidence (narration, notes, dates) supports treating this as one
genuine split payment, not to re-check the sum.

Ledger record:
{json.dumps(ledger_summary, indent=2)}

Candidate gateway legs (sum to {total} paise, matching the ledger amount):
{json.dumps(leg_summaries, indent=2)}

Respond with ONLY a JSON object, no markdown fences, no prose:
{{
  "action": "confirm" | "reject",
  "reason": "short, specific reason referencing the actual narration/dates/evidence"
}}

"confirm" means the evidence supports this being one real split payment.
"reject" means something looks off despite the amounts summing correctly.
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


def verify_subset_llm(cluster_id, leg_ids, ledger_by_id, gateway_by_id, cluster, llm_call_fn=None):
    cached = _load_cached(cluster_id, leg_ids)
    if cached:
        return cached

    ledger_id = cluster["member_record_ids"][0]
    ledger_summary = _record_summary(ledger_by_id[ledger_id], "ledger")
    leg_summaries = [
        {"record_id": lid, **_record_summary(gateway_by_id[lid], "gateway")} for lid in leg_ids
    ]
    total = sum(gateway_by_id[lid]["amount"] for lid in leg_ids)
    prompt = _build_prompt(ledger_summary, leg_summaries, total)

    call = llm_call_fn or _call_groq
    raw = call(prompt)
    decision = _parse_response(raw)
    _save_cache(cluster_id, leg_ids, decision)
    return decision


def run_transitive_link(clusters, dataset, decision_log, llm_call_fn=None):
    ledger_by_id = {r["record_id"]: r for r in dataset["ledger"]}
    gateway_by_id = {r["record_id"]: r for r in dataset["gateway"]}

    def log(cluster_id, decision, reason):
        decision_log.setdefault(cluster_id, []).append({
            "agent": "transitive_link", "decision": decision, "reason": reason,
        })

    candidates = [
        (cid, c) for cid, c in clusters.items()
        if c["status"] == "unresolved_direct_match_exhausted"
    ]

    for cid, cluster in candidates:
        subsets = find_subset_sum_candidates(cluster, ledger_by_id, gateway_by_id)
        if not subsets:
            continue

        best_subset = subsets[0]
        try:
            decision = verify_subset_llm(cid, best_subset, ledger_by_id, gateway_by_id, cluster, llm_call_fn)
        except Exception as e:
            cluster["status"] = "unresolved_system_error"
            log(cid, "unresolved_system_error", f"API failure: {e}")
            continue

        if decision.get("action") == "confirm":
            cluster["status"] = "confirmed"
            cluster["chosen_gateway_record_ids"] = best_subset
            cluster["member_record_ids"] = cluster["member_record_ids"] + [
                lid for lid in best_subset if lid not in cluster["member_record_ids"]
            ]
            if "gateway" not in cluster["source_coverage"]:
                cluster["source_coverage"].append("gateway")
            log(cid, "confirmed", decision.get("reason", ""))
        else:
            log(cid, "not_confirmed", decision.get("reason", ""))

    return clusters
