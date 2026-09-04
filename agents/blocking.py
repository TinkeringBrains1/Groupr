"""
Blocking -- candidate narrowing, runs after Duplicate check, before Cluster
Agent. Rule-based, no LLM calls. Anchored on Ledger: order_id fast-path for
Gateway candidates (amount+date fallback if missing/garbled), settlement_id
lookup through matched gateway candidates for Settlement candidates
(amount+date fallback ONLY when zero gateway candidates exist -- catches
standalone phantom settlements).
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import config


def _within_tolerance(amount_a, amount_b, date_a, date_b, date_tolerance_days=None):
    tolerance = date_tolerance_days if date_tolerance_days is not None else config.BLOCKING_DATE_TOLERANCE_DAYS
    amount_ok = abs(amount_a - amount_b) <= config.BLOCKING_AMOUNT_TOLERANCE_PAISE
    date_ok = abs((date_a - date_b).days) <= tolerance
    return amount_ok and date_ok


def _find_gateway_candidates(ledger_rec, gateway_records, gateway_by_order_id):
    order_id = ledger_rec["order_id"]
    matches = gateway_by_order_id.get(order_id)
    if matches:
        return [{"record_id": g["record_id"], "match_method": "order_id"} for g in matches]

    candidates = []
    for g in gateway_records:
        if _within_tolerance(g["amount"], ledger_rec["amount"], g["created_at"], ledger_rec["booked_date"]):
            candidates.append({"record_id": g["record_id"], "match_method": "amount_date_fallback"})
    return candidates


def _find_settlement_candidates(ledger_rec, gateway_candidate_ids, gateway_by_id, settlement_records):
    settlement_ids = set()
    for gid in gateway_candidate_ids:
        sid = gateway_by_id[gid].get("settlement_id")
        if sid:
            settlement_ids.add(sid)

    if settlement_ids:
        return [{"record_id": sid, "match_method": "settlement_id"} for sid in sorted(settlement_ids)]

    if gateway_candidate_ids:
        # gateway candidate(s) exist but none have a settlement_id yet --
        # genuinely pending, NOT a phantom case. Correctly return nothing
        # rather than falling back to a blind amount+date search.
        return []

    candidates = []
    for s in settlement_records:
        if _within_tolerance(
            s["amount"], ledger_rec["amount"], s["created_at"], ledger_rec["booked_date"],
            date_tolerance_days=config.BLOCKING_SETTLEMENT_FALLBACK_DATE_TOLERANCE_DAYS,
        ):
            candidates.append({"record_id": s["record_id"], "match_method": "amount_date_fallback"})
    return candidates


def run_blocking(clean_gateway, clean_ledger, settlement_records):
    gateway_by_id = {g["record_id"]: g for g in clean_gateway}
    gateway_by_order_id = {}
    for g in clean_gateway:
        gateway_by_order_id.setdefault(g["order_id"], []).append(g)

    candidate_pools = {}
    for ledger_rec in clean_ledger:
        gw_candidates = _find_gateway_candidates(ledger_rec, clean_gateway, gateway_by_order_id)
        gw_candidate_ids = [c["record_id"] for c in gw_candidates]
        settle_candidates = _find_settlement_candidates(
            ledger_rec, gw_candidate_ids, gateway_by_id, settlement_records
        )
        candidate_pools[ledger_rec["record_id"]] = {
            "gateway_candidates": gw_candidates,
            "settlement_candidates": settle_candidates,
        }

    return candidate_pools
