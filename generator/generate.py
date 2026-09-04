"""
Synthetic reconciliation dataset generator.

Run from anywhere:
    python generator/generate.py
or:
    cd generator && python generate.py
"""

import csv
import json
import os
import random
import sys
from datetime import datetime, timedelta

from faker import Faker

# Self-resolving import: works regardless of the caller's working directory.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import config
from narration import (
    generate_bank_narration,
    generate_ledger_narration,
    generate_gateway_notes,
)

fake = Faker("en_IN")
Faker.seed(config.SEED)

TODAY = datetime(2026, 8, 24)

gateway_records = []
settlement_records = []
ledger_records = []
ground_truth = []
settlement_plan = []
settlement_batches = []
_pending_gt_index = {}


def random_amount_paise():
    return random.randint(config.MIN_AMOUNT_PAISE, config.MAX_AMOUNT_PAISE)


def random_base_date():
    offset = random.randint(config.MIN_BASE_OFFSET_DAYS, config.MIN_BASE_OFFSET_DAYS + config.DATE_RANGE_DAYS)
    hour = random.randint(8, 22)
    minute = random.randint(0, 59)
    return TODAY - timedelta(days=offset, hours=-hour, minutes=-minute)


def pick_method():
    return random.choices(config.PAYMENT_METHODS, weights=config.METHOD_WEIGHTS, k=1)[0]


def settlement_offset(method):
    lo, hi = config.SETTLEMENT_WINDOW_DAYS[method]
    return timedelta(days=random.randint(lo, hi), hours=random.randint(0, 12))


def _random_hex(n=14):
    return "".join(random.choices("0123456789abcdef", k=n))


def new_order_id():
    return "ORD" + "".join(random.choices("0123456789", k=8))


def new_payment_id():
    return "pay_" + _random_hex(14)


def new_settlement_id():
    return "setl_" + _random_hex(14)


def new_ledger_id():
    return "LEDG" + "".join(random.choices("0123456789", k=8))


def new_rrn():
    return "".join(random.choices("0123456789", k=12))


def new_utr():
    return "".join(random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=16))


def counterparty_name():
    return fake.company() if random.random() < 0.5 else fake.name()


def bank_or_vpa(method):
    if method == "upi":
        return fake.user_name() + "@" + random.choice(["okhdfcbank", "ybl", "oksbi", "paytm"])
    return random.choice(["HDFC0001234", "ICIC0000456", "SBIN0011223", "UTIB0002234"])


def ground_truth_record(record_id, source, txn_id):
    _pending_gt_index[record_id] = {"record_id": record_id, "source": source, "true_transaction_id": txn_id}


def make_gateway_record(
    order_id, amount_paise, created_at, method, txn_id,
    include_order_ref=True, status="captured",
    settle=True, settle_target_date=None, settle_contribute_amount=None,
):
    rec_id = new_payment_id()
    rec = {
        "record_id": rec_id, "payment_id": rec_id, "order_id": order_id,
        "amount": amount_paise, "currency": "INR", "status": status, "method": method,
        "vpa_or_bank": bank_or_vpa(method), "email": fake.email(), "contact": fake.phone_number(),
        "acquirer_rrn": new_rrn(),
        "notes": json.dumps(generate_gateway_notes(order_id, include_ref=include_order_ref)),
        "created_at": created_at.isoformat(), "settlement_id": "",
    }
    gateway_records.append(rec)
    ground_truth_record(rec_id, "gateway", txn_id)

    if settle:
        target_date = settle_target_date or (created_at + settlement_offset(method))
        contribute_amount = settle_contribute_amount if settle_contribute_amount is not None else amount_paise
        settlement_plan.append({
            "gateway_record_id": rec_id, "contribute_amount": contribute_amount,
            "target_date": target_date, "method": method,
        })
    return rec


def make_settlement_record(amount_paise, created_at, txn_id=None, utr=None):
    rec_id = new_settlement_id()
    utr = utr or new_utr()
    rec = {
        "record_id": rec_id, "settlement_id": rec_id, "amount": amount_paise, "utr": utr,
        "status": "processed", "narration": generate_bank_narration(utr=utr),
        "created_at": created_at.isoformat(),
    }
    settlement_records.append(rec)
    ground_truth_record(rec_id, "settlement", txn_id)
    return rec


def make_ledger_record(order_id, amount_paise, booked_date, counterparty, txn_id, kind="clean", total=None):
    rec_id = new_ledger_id()
    rec = {
        "record_id": rec_id, "ledger_entry_id": rec_id, "order_id": order_id,
        "amount": amount_paise, "booked_date": booked_date.isoformat(),
        "account_head": "Accounts Receivable", "counterparty": counterparty,
        "narration": generate_ledger_narration(order_id, counterparty, kind=kind, total=total),
    }
    ledger_records.append(rec)
    ground_truth_record(rec_id, "ledger", txn_id)
    return rec


def build_clean(txn_id):
    order_id = new_order_id()
    amount = random_amount_paise()
    base_date = random_base_date()
    method = pick_method()
    counterparty = counterparty_name()
    make_gateway_record(order_id, amount, base_date, method, txn_id)
    make_ledger_record(order_id, amount, base_date, counterparty, txn_id)
    return {"transaction_id": txn_id, "noise_type": "clean", "true_amount": amount}


def build_rounding(txn_id):
    order_id = new_order_id()
    amount = random_amount_paise()
    base_date = random_base_date()
    method = pick_method()
    counterparty = counterparty_name()
    noise = random.randint(*config.ROUNDING_NOISE_PAISE_RANGE)
    sign = random.choice([1, -1])
    settled_contribute = amount + sign * noise
    make_gateway_record(order_id, amount, base_date, method, txn_id, settle_contribute_amount=settled_contribute)
    make_ledger_record(order_id, amount, base_date, counterparty, txn_id)
    return {"transaction_id": txn_id, "noise_type": "rounding", "true_amount": amount,
            "rounding_delta_paise": sign * noise}


def build_date_lag(txn_id):
    order_id = new_order_id()
    amount = random_amount_paise()
    base_date = random_base_date()
    method = pick_method()
    counterparty = counterparty_name()
    extra_lag = timedelta(days=random.randint(*config.DATE_LAG_DAYS_RANGE))
    delayed_target = base_date + settlement_offset(method) + extra_lag
    make_gateway_record(order_id, amount, base_date, method, txn_id, settle_target_date=delayed_target)
    make_ledger_record(order_id, amount, base_date, counterparty, txn_id)
    return {"transaction_id": txn_id, "noise_type": "date_lag", "true_amount": amount,
            "extra_lag_days": extra_lag.days}


def build_split_payment(txn_id, subpattern):
    order_id = new_order_id()
    total_amount = random_amount_paise()
    base_date = random_base_date()
    counterparty = counterparty_name()
    num_legs = random.choice([2, 2, 3])
    fractions = sorted([random.random() for _ in range(num_legs - 1)])
    fractions = [0] + fractions + [1]
    leg_amounts = [int(round((fractions[i + 1] - fractions[i]) * total_amount)) for i in range(num_legs)]
    leg_amounts[-1] += total_amount - sum(leg_amounts)

    leg_dates, leg_methods = [], []
    for i in range(num_legs):
        method = pick_method()
        leg_methods.append(method)
        if subpattern == "same_session":
            leg_dates.append(base_date + timedelta(minutes=random.randint(0, 45) * i))
        else:
            leg_dates.append(base_date + timedelta(days=random.randint(2, 10) * i))

    for i in range(num_legs):
        make_gateway_record(order_id, leg_amounts[i], leg_dates[i], leg_methods[i], txn_id)

    make_ledger_record(order_id, total_amount, base_date, counterparty, txn_id, kind="split", total=num_legs)
    return {"transaction_id": txn_id, "noise_type": "split_payment", "true_amount": total_amount,
            "split_subpattern": subpattern, "num_legs": num_legs, "leg_amounts": leg_amounts}


def build_missing_counterpart(txn_id, difficulty):
    order_id = new_order_id()
    amount = random_amount_paise()
    base_date = random_base_date()
    method = pick_method()
    counterparty = counterparty_name()
    missing_source = random.choice(["gateway", "settlement", "ledger"])

    if missing_source == "gateway":
        make_settlement_record(amount, base_date + settlement_offset(method), txn_id=txn_id)
        make_ledger_record(order_id, amount, base_date, counterparty, txn_id, kind=difficulty)
    elif missing_source == "settlement":
        make_gateway_record(order_id, amount, base_date, method, txn_id, settle=False)
        make_ledger_record(order_id, amount, base_date, counterparty, txn_id, kind=difficulty)
    else:
        make_gateway_record(order_id, amount, base_date, method, txn_id)

    return {"transaction_id": txn_id, "noise_type": "missing_counterpart", "true_amount": amount,
            "missing_source": missing_source, "difficulty": difficulty}


def build_duplicate(txn_id):
    order_id = new_order_id()
    amount = random_amount_paise()
    base_date = random_base_date()
    method = pick_method()
    counterparty = counterparty_name()
    make_gateway_record(order_id, amount, base_date, method, txn_id)
    make_ledger_record(order_id, amount, base_date, counterparty, txn_id)

    dup_source = random.choice(["gateway", "ledger"])
    exact = random.random() > config.DUPLICATE_REFERENCE_MISMATCH_RATE
    dup_order_id = order_id if exact else order_id[:-2] + "XX"

    if dup_source == "gateway":
        make_gateway_record(dup_order_id, amount, base_date, method, txn_id, include_order_ref=exact)
    else:
        make_ledger_record(dup_order_id, amount, base_date, counterparty, txn_id)

    return {"transaction_id": txn_id, "noise_type": "duplicate", "true_amount": amount,
            "duplicate_source": dup_source, "exact_duplicate": exact}


def run_settlement_batching():
    from collections import defaultdict
    buckets = defaultdict(list)
    for entry in settlement_plan:
        key = (entry["target_date"].date().isoformat(), entry["method"])
        buckets[key].append(entry)

    gateway_by_id = {r["record_id"]: r for r in gateway_records}
    for (date_str, method), entries in sorted(buckets.items()):
        batch_amount = sum(e["contribute_amount"] for e in entries)
        settle_dt = datetime.fromisoformat(date_str).replace(hour=config.SETTLEMENT_BATCH_HOUR)
        rec = make_settlement_record(batch_amount, settle_dt, txn_id=None)
        member_ids = [e["gateway_record_id"] for e in entries]
        for gid in member_ids:
            gateway_by_id[gid]["settlement_id"] = rec["settlement_id"]
        settlement_batches.append({
            "settlement_record_id": rec["record_id"], "method": method,
            "batch_date": date_str, "gateway_record_ids": member_ids,
        })


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    txn_counter = 0

    def next_txn_id():
        nonlocal txn_counter
        txn_counter += 1
        return f"TXN{txn_counter:04d}"

    for _ in range(config.NOISE_COUNTS["clean"]):
        ground_truth.append(build_clean(next_txn_id()))
    for _ in range(config.NOISE_COUNTS["rounding"]):
        ground_truth.append(build_rounding(next_txn_id()))
    for _ in range(config.NOISE_COUNTS["date_lag"]):
        ground_truth.append(build_date_lag(next_txn_id()))
    for _ in range(config.SPLIT_SUBPATTERN["same_session"]):
        ground_truth.append(build_split_payment(next_txn_id(), "same_session"))
    for _ in range(config.SPLIT_SUBPATTERN["staged_deferred"]):
        ground_truth.append(build_split_payment(next_txn_id(), "staged_deferred"))
    for _ in range(config.MISSING_DIFFICULTY["obvious_error"]):
        ground_truth.append(build_missing_counterpart(next_txn_id(), "obvious_error"))
    for _ in range(config.MISSING_DIFFICULTY["ambiguous"]):
        ground_truth.append(build_missing_counterpart(next_txn_id(), "ambiguous"))
    for _ in range(config.NOISE_COUNTS["duplicate"]):
        ground_truth.append(build_duplicate(next_txn_id()))

    run_settlement_batching()

    random.shuffle(gateway_records)
    random.shuffle(settlement_records)
    random.shuffle(ledger_records)

    def write_csv(path, records):
        if not records:
            return
        fieldnames = list(records[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    write_csv(os.path.join(config.OUTPUT_DIR, "gateway.csv"), gateway_records)
    write_csv(os.path.join(config.OUTPUT_DIR, "settlement.csv"), settlement_records)
    write_csv(os.path.join(config.OUTPUT_DIR, "ledger.csv"), ledger_records)

    gt_out = {
        "transactions": ground_truth,
        "record_index": list(_pending_gt_index.values()),
        "settlement_batches": settlement_batches,
    }
    with open(os.path.join(config.OUTPUT_DIR, "ground_truth.json"), "w", encoding="utf-8") as f:
        json.dump(gt_out, f, indent=2)

    pending_count = sum(1 for r in gateway_records if not r["settlement_id"])
    batch_sizes = [len(b["gateway_record_ids"]) for b in settlement_batches]
    avg_batch = round(sum(batch_sizes) / len(batch_sizes), 1) if batch_sizes else 0

    summary_lines = [
        "Synthetic reconciliation dataset -- generation summary",
        "=" * 55,
        f"True transactions: {len(ground_truth)}",
        f"Gateway records:    {len(gateway_records)}",
        f"Settlement records: {len(settlement_records)}  ({len(settlement_batches)} real batches "
        f"+ {len(settlement_records) - len(settlement_batches)} phantom/standalone)",
        f"Ledger records:     {len(ledger_records)}",
        f"Total raw records:  {len(gateway_records) + len(settlement_records) + len(ledger_records)}",
        "",
        f"Gateway payments with a settlement_id assigned: {len(gateway_records) - pending_count} / {len(gateway_records)}",
        f"Gateway payments still pending (no settlement):  {pending_count}",
        f"Average settlement batch size: {avg_batch} payments/batch",
        "",
        "Noise category counts:",
    ]
    for k, v in config.NOISE_COUNTS.items():
        summary_lines.append(f"  {k:22s} {v}")
    summary_lines += [
        "",
        f"Split subpatterns:  same_session={config.SPLIT_SUBPATTERN['same_session']}, "
        f"staged_deferred={config.SPLIT_SUBPATTERN['staged_deferred']}",
        f"Missing difficulty: obvious_error={config.MISSING_DIFFICULTY['obvious_error']}, "
        f"ambiguous={config.MISSING_DIFFICULTY['ambiguous']}",
    ]
    summary_text = "\n".join(summary_lines)
    with open(os.path.join(config.OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_text)
    print(summary_text)


if __name__ == "__main__":
    main()
