"""
Loads the generated dataset (gateway.csv, settlement.csv, ledger.csv) into
typed Python records -- amounts as int (paise), dates as datetime.
"""

import csv
import json
import os
from datetime import datetime


def _parse_gateway(row):
    row = dict(row)
    row["amount"] = int(row["amount"])
    row["created_at"] = datetime.fromisoformat(row["created_at"])
    return row


def _parse_settlement(row):
    row = dict(row)
    row["amount"] = int(row["amount"])
    row["created_at"] = datetime.fromisoformat(row["created_at"])
    return row


def _parse_ledger(row):
    row = dict(row)
    row["amount"] = int(row["amount"])
    row["booked_date"] = datetime.fromisoformat(row["booked_date"])
    return row


def load_dataset(output_dir):
    with open(os.path.join(output_dir, "gateway.csv")) as f:
        gateway = [_parse_gateway(r) for r in csv.DictReader(f)]
    with open(os.path.join(output_dir, "settlement.csv")) as f:
        settlement = [_parse_settlement(r) for r in csv.DictReader(f)]
    with open(os.path.join(output_dir, "ledger.csv")) as f:
        ledger = [_parse_ledger(r) for r in csv.DictReader(f)]
    return {"gateway": gateway, "settlement": settlement, "ledger": ledger}


def load_ground_truth(output_dir):
    """Dev/eval use only -- never fed into the reconciliation pipeline itself."""
    with open(os.path.join(output_dir, "ground_truth.json")) as f:
        return json.load(f)


def index_by(records, key):
    return {r[key]: r for r in records}


def group_by(records, key):
    out = {}
    for r in records:
        out.setdefault(r[key], []).append(r)
    return out
