"""
Settlement Q&A Agent -- covers the "Settlement Q&A agent" example direction
from the track brief. A THIN layer over the FINISHED pipeline output --
queries clusters/decision_log/eval_report.json directly, never re-runs any
agent, never calls Groq, and never touches ground_truth.json.

Deliberately narrow: exactly 5 supported question types, template-matched
by keyword/pattern, not open-domain RAG.

Supported questions:
  1. "Why wasn't [order_id / record_id] matched?"
  2. "Show me all split payments"
  3. "What's our match rate / false positive rate?"
  4. "Show exceptions of type X" (X = duplicate / phantom / split / all)
  5. "Why is [cluster_id]'s confidence low?"

Usage:
    python evaluation/qa_agent.py                  # interactive prompt
    python evaluation/qa_agent.py "your question"   # one-shot
"""

import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "agents"))
import io_utils

OUTPUT_DIR = os.path.join(_ROOT, "output")


def load_all():
    with open(os.path.join(OUTPUT_DIR, "final_result.json")) as f:
        result = json.load(f)
    with open(os.path.join(OUTPUT_DIR, "eval_report.json")) as f:
        eval_report = json.load(f)
    ds = io_utils.load_dataset(OUTPUT_DIR)
    return result, eval_report, ds


def build_indexes(result, ds):
    clusters = result["clusters"]
    ledger_by_id = {r["record_id"]: r for r in ds["ledger"]}
    gateway_by_id = {r["record_id"]: r for r in ds["gateway"]}

    order_id_to_cluster = {}
    for cid, c in clusters.items():
        ledger_id = c["member_record_ids"][0]
        if ledger_id in ledger_by_id:
            order_id_to_cluster[ledger_by_id[ledger_id]["order_id"]] = cid

    return clusters, ledger_by_id, gateway_by_id, order_id_to_cluster


ID_PATTERN = re.compile(r"\b(ORD\d+|LEDG\d+|CLUS_\w+|pay_\w+|setl_\w+)\b")


def extract_id(question):
    m = ID_PATTERN.search(question)
    return m.group(1) if m else None


def classify(question):
    q = question.lower()
    if ("why" in q) and ("confidence" in q):
        return "confidence"
    has_negation = bool(re.search(r"\bnot\b", q)) or "n't" in q
    if has_negation and "match" in q:
        return "why_not_matched"
    if "split payment" in q:
        return "show_splits"
    if "match rate" in q or "false positive" in q:
        return "rates"
    if "exception" in q:
        return "show_exceptions"
    return "unsupported"


def handle_why_not_matched(question, result, clusters, ledger_by_id, order_id_to_cluster):
    rid = extract_id(question)
    if not rid:
        return "Please include an order_id (e.g. ORD12345678) or cluster_id in your question."

    cid = rid if rid.startswith("CLUS_") else order_id_to_cluster.get(rid)
    if not cid or cid not in clusters:
        return f"No record found for '{rid}'."

    c = clusters[cid]
    history = result["decision_log"].get(cid, [])
    if c["status"] == "confirmed":
        return f"{cid} WAS matched (status: confirmed). It didn't fail -- did you mean to ask why it succeeded?"

    lines = [f"{cid} (status: {c['status']}):"]
    for h in history:
        lines.append(f"  [{h['agent']}] {h['decision']}: {h['reason']}")
    if not history:
        lines.append("  (no decision history logged)")
    return "\n".join(lines)


def handle_show_splits(clusters, ledger_by_id):
    splits = [
        (cid, c) for cid, c in clusters.items()
        if c.get("chosen_gateway_record_ids") and len(c["chosen_gateway_record_ids"]) > 1
    ]
    if not splits:
        return "No split payments resolved by the system."
    lines = [f"Found {len(splits)} resolved split payments:"]
    for cid, c in splits:
        ledger_id = c["member_record_ids"][0]
        order_id = ledger_by_id.get(ledger_id, {}).get("order_id", "?")
        n_legs = len(c["chosen_gateway_record_ids"])
        lines.append(f"  {order_id} ({cid}): {n_legs} legs")
    return "\n".join(lines)


def handle_rates(eval_report):
    return (
        f"Match rate: {eval_report['match_rate']*100:.1f}% "
        f"({eval_report['correct_count']}/{eval_report['total_clusters']})\n"
        f"False positive rate: {eval_report['false_positive_rate']*100:.1f}% "
        f"({eval_report['wrong_count']}/{eval_report['confirmed_count']} confirmed)"
    )


def handle_show_exceptions(question, eval_report):
    q = question.lower()
    exceptions = eval_report["exceptions"]

    if "duplicate" in q:
        filtered = [e for e in exceptions if "duplicate" in e["status"]]
        label = "duplicate"
    elif "phantom" in q or "gateway" in q:
        filtered = [e for e in exceptions if "settlement record present" in e["reason"].lower()]
        label = "phantom-settlement"
    elif "split" in q:
        filtered = [e for e in exceptions if "split" in e["reason"].lower() or "legs" in e["reason"].lower()]
        label = "split-payment"
    else:
        filtered = exceptions
        label = "all"

    if not filtered:
        return f"No {label} exceptions found."
    lines = [f"{len(filtered)} {label} exception(s):"]
    for e in filtered:
        lines.append(f"  [{e['status']}] {e['ledger_order_id']}: {e['reason'][:100]}")
    return "\n".join(lines)


def handle_confidence(question, clusters, ledger_by_id, order_id_to_cluster):
    rid = extract_id(question)
    if not rid:
        return "Please include an order_id or cluster_id in your question."
    cid = rid if rid.startswith("CLUS_") else order_id_to_cluster.get(rid)
    if not cid or cid not in clusters:
        return f"No record found for '{rid}'."

    c = clusters[cid]
    conf = c.get("confidence")
    gw_methods = [x["match_method"] for x in c.get("gateway_candidates", [])]
    settle_methods = [x["match_method"] for x in c.get("settlement_candidates", [])]
    return (
        f"{cid} confidence: {conf}\n"
        f"  Gateway candidates found via: {gw_methods or 'none'}\n"
        f"  Settlement candidates found via: {settle_methods or 'none'}\n"
        f"  (order_id/settlement_id = reliable match; amount_date_fallback = a guess, "
        f"which pulls confidence down)"
    )


def answer(question, result, eval_report, ds):
    clusters, ledger_by_id, gateway_by_id, order_id_to_cluster = build_indexes(result, ds)
    intent = classify(question)

    if intent == "why_not_matched":
        return handle_why_not_matched(question, result, clusters, ledger_by_id, order_id_to_cluster)
    if intent == "show_splits":
        return handle_show_splits(clusters, ledger_by_id)
    if intent == "rates":
        return handle_rates(eval_report)
    if intent == "show_exceptions":
        return handle_show_exceptions(question, eval_report)
    if intent == "confidence":
        return handle_confidence(question, clusters, ledger_by_id, order_id_to_cluster)

    return (
        "I can only answer these types of questions:\n"
        "  - Why wasn't [order_id] matched?\n"
        "  - Show me all split payments\n"
        "  - What's our match rate / false positive rate?\n"
        "  - Show exceptions of type [duplicate/phantom/split/all]\n"
        "  - Why is [order_id]'s confidence low?"
    )


if __name__ == "__main__":
    result, eval_report, ds = load_all()

    if len(sys.argv) > 1:
        print(answer(" ".join(sys.argv[1:]), result, eval_report, ds))
    else:
        print("Settlement Q&A -- ask a question (or 'quit'):")
        while True:
            try:
                q = input("> ")
            except EOFError:
                break
            if q.strip().lower() in ("quit", "exit"):
                break
            print(answer(q, result, eval_report, ds))
            print()
