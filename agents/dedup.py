"""
Duplicate check -- runs before Blocking/Cluster Agent. Rule-based,
same-source only (gateway + ledger; settlement can't structurally duplicate
under the batch model). No amount/date tolerance -- a duplicate is a literal
same-amount, same-calendar-day copy, not a fuzzy match.
"""

import os
import re
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# Real order_ids are always "ORD" + 8 digits. A garbled duplicate reference
# breaks this pattern, letting us prefer keeping the well-formed original
# over the garbled copy when choosing which record survives a duplicate pair.
_ORDER_ID_PATTERN = re.compile(r"^ORD\d{8}$")


def _is_well_formed_order_id(ref):
    return bool(_ORDER_ID_PATTERN.match(ref))


def _find_duplicates_in_source(records, ref_field):
    date_field = "created_at" if "created_at" in records[0] else "booked_date"
    groups = defaultdict(list)
    for r in records:
        key = (r["amount"], r[date_field].date())
        groups[key].append(r)

    exact_duplicate_ids = []
    possible_duplicates = []

    for (amount, date), group in groups.items():
        if len(group) < 2:
            continue

        by_ref = defaultdict(list)
        for r in group:
            by_ref[r[ref_field]].append(r)

        for ref, recs in by_ref.items():
            if len(recs) > 1:
                original = recs[0]
                for dup in recs[1:]:
                    exact_duplicate_ids.append({
                        "record_id": dup["record_id"], "duplicate_of": original["record_id"],
                        "reason": f"identical amount (Rs{amount/100:.2f}), date, and reference ({ref}) as {original['record_id']}",
                    })

        distinct_refs = list(by_ref.keys())
        if len(distinct_refs) > 1:
            well_formed_refs = [ref for ref in distinct_refs if _is_well_formed_order_id(ref)]
            anchor_ref = well_formed_refs[0] if well_formed_refs else distinct_refs[0]
            anchor = by_ref[anchor_ref][0]
            for ref in distinct_refs:
                if ref == anchor_ref:
                    continue
                for r in by_ref[ref]:
                    possible_duplicates.append({
                        "record_id": r["record_id"], "possible_duplicate_of": anchor["record_id"],
                        "reason": f"same amount (Rs{amount/100:.2f}) and date as {anchor['record_id']}, "
                                  f"but reference differs ('{r[ref_field]}' vs '{anchor[ref_field]}')",
                    })

    return exact_duplicate_ids, possible_duplicates


def run_duplicate_check(dataset):
    gw_exact, gw_possible = _find_duplicates_in_source(dataset["gateway"], "order_id")
    lg_exact, lg_possible = _find_duplicates_in_source(dataset["ledger"], "order_id")

    exact_ids = {d["record_id"] for d in gw_exact} | {d["record_id"] for d in lg_exact}
    possible_ids = {d["record_id"] for d in gw_possible} | {d["record_id"] for d in lg_possible}
    removed_ids = exact_ids | possible_ids

    clean_gateway = [r for r in dataset["gateway"] if r["record_id"] not in removed_ids]
    clean_ledger = [r for r in dataset["ledger"] if r["record_id"] not in removed_ids]

    exception_report = []
    for d in gw_exact:
        exception_report.append({**d, "source": "gateway", "status": "skipped_duplicate_exact"})
    for d in lg_exact:
        exception_report.append({**d, "source": "ledger", "status": "skipped_duplicate_exact"})
    for d in gw_possible:
        exception_report.append({**d, "source": "gateway", "status": "flagged_possible_duplicate"})
    for d in lg_possible:
        exception_report.append({**d, "source": "ledger", "status": "flagged_possible_duplicate"})

    return {
        "clean_gateway": clean_gateway,
        "clean_ledger": clean_ledger,
        "exception_report": exception_report,
    }
