"""
Evaluator -- standalone scoring module, NOT a pipeline stage. Reads the
final saved pipeline state and joins it against the hidden ground truth
(which the pipeline itself never sees). No Groq calls, no agent logic.

Run from anywhere:
    python evaluation/scorer.py
"""

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "agents"))

OUTPUT_DIR = os.path.join(_ROOT, "output")


def load_final_result(path=None):
    path = path or os.path.join(OUTPUT_DIR, "final_result.json")
    with open(path) as f:
        return json.load(f)


def load_ground_truth(path=None):
    path = path or os.path.join(OUTPUT_DIR, "ground_truth.json")
    with open(path) as f:
        return json.load(f)


def build_lookups(gt):
    txn_by_id = {t["transaction_id"]: t for t in gt["transactions"]}
    record_to_txn = {r["record_id"]: r["true_transaction_id"] for r in gt["record_index"]}
    return txn_by_id, record_to_txn


def noise_label(txn):
    label = txn.get("noise_type", "unknown")
    if label == "missing_counterpart":
        label += f"/{txn.get('missing_source')}"
    return label


def get_chosen_gateway_ids(cluster):
    if cluster.get("chosen_gateway_record_ids"):
        return list(cluster["chosen_gateway_record_ids"])
    if cluster.get("chosen_gateway_record_id"):
        return [cluster["chosen_gateway_record_id"]]
    return []


def is_cluster_correct(cluster, record_to_txn):
    """True (correct) / False (false positive) / None (unverifiable)."""
    ledger_id = cluster["member_record_ids"][0]
    true_txn = record_to_txn.get(ledger_id)

    gw_ids = get_chosen_gateway_ids(cluster)
    settle_id = cluster.get("chosen_settlement_record_id")

    if gw_ids:
        return all(record_to_txn.get(gid) == true_txn for gid in gw_ids)

    if settle_id:
        settle_txn = record_to_txn.get(settle_id)
        if settle_txn is None:
            return None  # batch settlement, can't verify this way
        return settle_txn == true_txn

    return None


def three_sources_present(cluster):
    return bool(get_chosen_gateway_ids(cluster)) and bool(cluster.get("chosen_settlement_record_id"))


def is_exact_amount(cluster, ledger_by_id, gateway_by_id, settlement_by_id):
    ledger_id = cluster["member_record_ids"][0]
    ledger_amt = ledger_by_id[ledger_id]["amount"]
    gw_ids = get_chosen_gateway_ids(cluster)
    if gw_ids:
        return sum(gateway_by_id[g]["amount"] for g in gw_ids if g in gateway_by_id) == ledger_amt
    settle_id = cluster.get("chosen_settlement_record_id")
    if settle_id and settle_id in settlement_by_id:
        return settlement_by_id[settle_id]["amount"] == ledger_amt
    return True


def classify_cluster(cluster, ledger_by_id, gateway_by_id, settlement_by_id):
    status = cluster["status"]
    if status == "flagged_impossible":
        return "skipped"
    if status != "confirmed":
        return "exception"
    if three_sources_present(cluster):
        exact = is_exact_amount(cluster, ledger_by_id, gateway_by_id, settlement_by_id)
        return "matched" if exact else "partially_matched"
    return "one_sided"


def score(output_dir=None):
    output_dir = output_dir or OUTPUT_DIR
    data = load_final_result(os.path.join(output_dir, "final_result.json"))
    gt = load_ground_truth(os.path.join(output_dir, "ground_truth.json"))
    txn_by_id, record_to_txn = build_lookups(gt)

    import io_utils
    ds = io_utils.load_dataset(output_dir)
    ledger_by_id = {r["record_id"]: r for r in ds["ledger"]}
    gateway_by_id = {r["record_id"]: r for r in ds["gateway"]}
    settlement_by_id = {r["record_id"]: r for r in ds["settlement"]}

    clusters = data["clusters"]
    decision_log = data["decision_log"]
    dedup_exceptions = data.get("dedup_exception_report", [])

    confirmed = [c for c in clusters.values() if c["status"] == "confirmed"]
    correct, wrong, unverifiable = [], [], []
    for c in confirmed:
        result = is_cluster_correct(c, record_to_txn)
        (correct if result is True else wrong if result is False else unverifiable).append(c)

    total_clusters = len(clusters)
    match_rate = len(correct) / total_clusters if total_clusters else 0
    false_positive_rate = len(wrong) / len(confirmed) if confirmed else 0

    category_stats = {}
    for cid, c in clusters.items():
        ledger_id = c["member_record_ids"][0]
        txn = txn_by_id.get(record_to_txn.get(ledger_id), {})
        label = noise_label(txn)
        cat = category_stats.setdefault(label, {"total": 0, "correct": 0, "wrong": 0, "exception": 0, "unverifiable": 0})
        cat["total"] += 1
        if c["status"] == "confirmed":
            result = is_cluster_correct(c, record_to_txn)
            if result is True:
                cat["correct"] += 1
            elif result is False:
                cat["wrong"] += 1
            else:
                cat["unverifiable"] += 1
        else:
            cat["exception"] += 1

    exceptions = []
    for cid, c in clusters.items():
        if c["status"] != "confirmed":
            ledger_id = c["member_record_ids"][0]
            reasons = [e["reason"] for e in decision_log.get(cid, []) if e["agent"] in ("drift", "direct_match", "transitive_link")]
            exceptions.append({
                "cluster_id": cid, "status": c["status"],
                "ledger_order_id": ledger_by_id.get(ledger_id, {}).get("order_id", "?"),
                "reason": reasons[-1] if reasons else "(no reason logged)",
            })
    for e in dedup_exceptions:
        exceptions.append({
            "cluster_id": None, "status": e["status"],
            "ledger_order_id": e.get("record_id", "?"), "reason": e["reason"],
        })

    report_categories = {"matched": 0, "partially_matched": 0, "one_sided": 0, "skipped": 0, "exception": 0}
    for c in clusters.values():
        report_categories[classify_cluster(c, ledger_by_id, gateway_by_id, settlement_by_id)] += 1
    report_categories["skipped"] += len(dedup_exceptions)

    decision_counts = {}
    for entries in decision_log.values():
        for e in entries:
            decision_counts[e["agent"]] = decision_counts.get(e["agent"], 0) + 1

    return {
        "total_clusters": total_clusters, "match_rate": match_rate,
        "false_positive_rate": false_positive_rate, "confirmed_count": len(confirmed),
        "correct_count": len(correct), "wrong_count": len(wrong),
        "unverifiable_count": len(unverifiable), "category_stats": category_stats,
        "report_categories": report_categories, "exceptions": exceptions,
        "decision_counts": decision_counts,
    }


def print_report(result):
    print("=" * 60)
    print("RECONCILIATION PIPELINE -- EVALUATION REPORT")
    print("=" * 60)
    print()
    print("--- Accuracy (side by side, no combined score) ---")
    print(f"  Match rate:          {result['match_rate']*100:.1f}%  "
          f"({result['correct_count']}/{result['total_clusters']} clusters correctly resolved)")
    print(f"  False positive rate: {result['false_positive_rate']*100:.1f}%  "
          f"({result['wrong_count']}/{result['confirmed_count']} confirmed clusters were wrong)")
    if result["unverifiable_count"]:
        print(f"  Unverifiable confirms: {result['unverifiable_count']} (should be 0)")
    print()
    print("--- Per-noise-category accuracy ---")
    for label, s in sorted(result["category_stats"].items()):
        print(f"  {label:32s} total={s['total']:3d}  correct={s['correct']:3d}  "
              f"wrong={s['wrong']:3d}  exception={s['exception']:3d}")
    print()
    print("--- Reporting categories ---")
    for cat, n in result["report_categories"].items():
        print(f"  {cat:20s} {n}")
    print()
    print(f"--- Honest exception list ({len(result['exceptions'])} items) ---")
    for e in result["exceptions"]:
        print(f"  [{e['status']}] {e['ledger_order_id']}: {e['reason'][:100]}")
    print()
    print("--- Throughput (decision-volume proxy) ---")
    for agent, n in result["decision_counts"].items():
        print(f"  {agent}: {n} decisions logged")


if __name__ == "__main__":
    result = score()
    print_report(result)
    with open(os.path.join(OUTPUT_DIR, "eval_report.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {os.path.join(OUTPUT_DIR, 'eval_report.json')}")
