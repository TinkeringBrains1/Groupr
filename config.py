"""
Config for the reconciliation synthetic dataset.
All numbers here match the locked architecture plan:
 - 150 true transactions -> ~450+ raw records across 3 sources
 - 6 noise categories, exact counts (not just %) so distribution is deterministic
 - method-aware settlement windows (real Razorpay-grounded SLAs)
"""

import os
import random

SEED = 42  # change this to get a different dataset; keep fixed for reproducibility
random.seed(SEED)

# ---- Dataset size & noise distribution (locked counts, sum to 150) ----
NOISE_COUNTS = {
    "clean": 60,
    "rounding": 20,
    "date_lag": 20,
    "split_payment": 20,
    "missing_counterpart": 20,
    "duplicate": 10,
}
assert sum(NOISE_COUNTS.values()) == 150

SPLIT_SUBPATTERN = {
    "same_session": 12,
    "staged_deferred": 8,
}
assert sum(SPLIT_SUBPATTERN.values()) == NOISE_COUNTS["split_payment"]

MISSING_DIFFICULTY = {
    "obvious_error": 10,
    "ambiguous": 10,
}
assert sum(MISSING_DIFFICULTY.values()) == NOISE_COUNTS["missing_counterpart"]

# ---- Payment methods & real-world-grounded settlement SLAs (in days) ----
PAYMENT_METHODS = ["upi", "card", "netbanking", "international"]
METHOD_WEIGHTS = [0.55, 0.25, 0.15, 0.05]

SETTLEMENT_WINDOW_DAYS = {
    "upi": (1, 1),
    "card": (2, 4),
    "netbanking": (2, 3),
    "international": (6, 9),
}

# ---- Amount & date ranges ----
MIN_AMOUNT_PAISE = 50_00
MAX_AMOUNT_PAISE = 250_000_00
DATE_RANGE_DAYS = 22            # compressed so real settlement batching (N:1) shows up meaningfully
MIN_BASE_OFFSET_DAYS = 30       # buffer so no record's date lands after the fixed TODAY anchor

# ---- Settlement batching ----
SETTLEMENT_BATCH_HOUR = 10

# ---- Noise tolerances ----
ROUNDING_NOISE_PAISE_RANGE = (1, 4000)      # Rs 0.01 to Rs 40
DATE_LAG_DAYS_RANGE = (1, 6)
DUPLICATE_REFERENCE_MISMATCH_RATE = 0.5

# ---- Blocking (candidate-search) tolerances ----
BLOCKING_AMOUNT_TOLERANCE_PAISE = 5000       # +/-Rs 50
BLOCKING_DATE_TOLERANCE_DAYS = 3
BLOCKING_SETTLEMENT_FALLBACK_DATE_TOLERANCE_DAYS = 12

# ---- Direct Match / Drift confirmation tolerance ----
CONFIRM_AMOUNT_TOLERANCE_PAISE = 500
CONFIRM_AMOUNT_TOLERANCE_PERCENT = 0.005

# ---- Output ----
# Resolved relative to THIS file's location, not the caller's working
# directory -- so "output/" always means repo-root/output regardless of
# where a script that imports this config was launched from.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
