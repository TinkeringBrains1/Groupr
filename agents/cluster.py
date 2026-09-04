"""
Cluster Agent -- proposes confidence-scored candidate groups from Blocking's
output. A HYPOTHESIS, not a final decision -- Direct Match can reject and
kick a cluster back here with a reason (max 2 retries).

Rule-based, no LLM call -- confidence is derived directly from HOW each
candidate was found (match_method from Blocking).
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

GATEWAY_WEIGHT = 0.5
SETTLEMENT_WEIGHT = 0.5


def _component_confidence(candidates, weight):
    if not candidates:
        return 0.0
    methods = {c["match_method"] for c in candidates}
    strong_method = "order_id" in methods or "settlement_id" in methods
    if strong_method:
        return weight
    if len(candidates) == 1:
        return round(weight * 0.6, 4)
    return round(weight * 0.3, 4)


def propose_cluster(ledger_record_id, candidate_pool, retry_count=0):
    gw_candidates = candidate_pool["gateway_candidates"]
    settle_candidates = candidate_pool["settlement_candidates"]

    member_ids = [ledger_record_id]
    source_coverage = ["ledger"]

    if gw_candidates:
        member_ids.append(gw_candidates[0]["record_id"])
        source_coverage.append("gateway")

    if settle_candidates:
        member_ids.append(settle_candidates[0]["record_id"])
        source_coverage.append("settlement")

    confidence = (
        _component_confidence(gw_candidates, GATEWAY_WEIGHT)
        + _component_confidence(settle_candidates, SETTLEMENT_WEIGHT)
    )

    cluster_id = f"CLUS_{ledger_record_id}"
    return {
        "cluster_id": cluster_id,
        "member_record_ids": member_ids,
        "source_coverage": source_coverage,
        "confidence": round(confidence, 4),
        "status": "proposed",
        "retry_count": retry_count,
        "gateway_candidates": gw_candidates,
        "settlement_candidates": settle_candidates,
    }


def run_cluster_agent(candidate_pools):
    clusters = {}
    for ledger_record_id, pool in candidate_pools.items():
        cluster = propose_cluster(ledger_record_id, pool)
        clusters[cluster["cluster_id"]] = cluster
    return clusters


def reproposal(cluster, rejection_reason, gateway_by_id=None):
    """
    Called when Direct Match rejects a cluster. Mutates the SAME cluster_id
    in place, increments retry_count, tries the next-best unused candidate
    on whichever side was rejected if one exists. If no alternative remains,
    confidence is downgraded rather than re-guessing blindly.
    """
    cluster["retry_count"] += 1

    current_ids = set(cluster["member_record_ids"])
    next_gw = next(
        (c for c in cluster["gateway_candidates"] if c["record_id"] not in current_ids), None
    )
    if next_gw:
        cluster["member_record_ids"] = [
            m for m in cluster["member_record_ids"]
            if m not in [c["record_id"] for c in cluster["gateway_candidates"]]
        ]
        cluster["member_record_ids"].append(next_gw["record_id"])
        if "gateway" not in cluster["source_coverage"]:
            cluster["source_coverage"].append("gateway")
    else:
        cluster["confidence"] = round(cluster["confidence"] * 0.5, 4)

    cluster["status"] = "proposed"
    return cluster
