"""
Runs the full reconciliation pipeline (via the LangGraph wrapper) against
real Groq calls, and saves the result to output/final_result.json -- the
input for evaluation/scorer.py and evaluation/qa_agent.py.

Uses each agent's disk cache automatically, so re-running after a partial
failure or during development costs near-zero new API calls.

Usage:
    python scripts/run_pipeline.py            # full batch
    python scripts/run_pipeline.py --limit 10  # small sample, for a quick check
"""

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "agents"))
OUTPUT_DIR = os.path.join(_ROOT, "output")

import io_utils
import graph as pipeline_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N ledger records (for a quick sample run).")
    args = parser.parse_args()

    ds = io_utils.load_dataset(OUTPUT_DIR)
    if args.limit:
        ds = dict(ds)
        ds["ledger"] = ds["ledger"][:args.limit]
        print(f"Running on a SAMPLE of {args.limit} ledger records "
              f"(gateway/settlement pools left full so real candidates are still found).")

    print(f"Running full graph pipeline on {len(ds['ledger'])} ledger records...\n")
    start = time.time()
    final_state = pipeline_graph.run_pipeline(ds)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s\n")

    clusters = final_state["clusters"]
    statuses = {}
    for c in clusters.values():
        statuses[c["status"]] = statuses.get(c["status"], 0) + 1
    print("=== Pipeline summary ===")
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count}")

    output = {
        "clusters": clusters,
        "decision_log": final_state["decision_log"],
        "dedup_exception_report": final_state["dedup_exception_report"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = os.path.join(OUTPUT_DIR, "final_result.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("Next: python evaluation/scorer.py")


if __name__ == "__main__":
    main()
