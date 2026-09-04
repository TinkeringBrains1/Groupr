"""
Narration / notes text generator.

Template-based, runs fully offline (no API key needed). If you want real
LLM-generated narration instead, flip USE_LLM=True and set GROQ_API_KEY --
generate_narration_llm() is provided as a documented hook, unused by default.
"""

import random

USE_LLM = False
GROQ_MODEL = "llama-3.1-8b-instant"


BANK_NARRATION_TEMPLATES = [
    "NEFT CR: {bank} {code} RAZORPAY SETTLEMENT",
    "IMPS CR-{code}-RAZORPAY SETTLEMENT-{bank}",
    "RTGS CR: {code} RAZORPAY PAYOUT {bank}",
    "UPI/{code}/RAZORPAY/SETTLEMENT",
    "NEFT-{bank}-{code}-RZRPY SETL",
]

BANKS = ["HDFC", "ICICI", "SBIN", "AXIS", "KKBK", "YESB", "IDFB"]


def _random_utr_code():
    return "".join(random.choices("0123456789abcdefghijklmnopqrstuvwxyz", k=16))


def generate_bank_narration(utr: str = None):
    template = random.choice(BANK_NARRATION_TEMPLATES)
    code = utr if utr else _random_utr_code()
    bank = random.choice(BANKS)
    return template.format(bank=bank, code=code)


LEDGER_NARRATION_TEMPLATES_CLEAN = [
    "Being amount received against Order {order_id} from {counterparty}",
    "Rcvd - {counterparty} - Order #{order_id}",
    "Order {order_id} settlement - {counterparty}",
    "Payment received - {counterparty} (Ord {order_id})",
]

LEDGER_NARRATION_TEMPLATES_SPLIT = [
    "Being full amount received against Order {order_id} from {counterparty}, across {total} payments",
    "Rcvd - {counterparty} - Order #{order_id} - settled via {total} separate payments",
    "Order {order_id} - {counterparty} - full value, split collection ({total} legs)",
    "Order {order_id} - {counterparty} - multi-payment settlement, {total} legs",
]

LEDGER_NARRATION_TEMPLATES_AMBIGUOUS = [
    "Being amount received against Order {order_id} from {counterparty} - UNRECONCILED",
    "Rcvd - {counterparty} - Order #{order_id} - pending bank confirmation",
    "Order {order_id} - {counterparty} - manual entry, source unclear",
]

LEDGER_NARRATION_TEMPLATES_OBVIOUS_ERROR = [
    "Order {order_id} - {counterparty} - TEST ENTRY, please reverse",
    "Rcvd - {counterparty} - Order #{order_id} - entered in error, appears duplicate of prior posting",
    "Order {order_id} - {counterparty} - wrong account posted, flagged for correction",
    "Being amount received against Order {order_id} from {counterparty} - MISMATCH flagged by ops, needs review",
]


def generate_ledger_narration(order_id, counterparty, kind="clean", leg=None, total=None):
    if kind == "split":
        template = random.choice(LEDGER_NARRATION_TEMPLATES_SPLIT)
        return template.format(order_id=order_id, counterparty=counterparty, total=total)
    if kind == "ambiguous":
        template = random.choice(LEDGER_NARRATION_TEMPLATES_AMBIGUOUS)
        return template.format(order_id=order_id, counterparty=counterparty)
    if kind == "obvious_error":
        template = random.choice(LEDGER_NARRATION_TEMPLATES_OBVIOUS_ERROR)
        return template.format(order_id=order_id, counterparty=counterparty)
    template = random.choice(LEDGER_NARRATION_TEMPLATES_CLEAN)
    return template.format(order_id=order_id, counterparty=counterparty)


def generate_gateway_notes(order_id, include_ref=True):
    if not include_ref:
        return {} if random.random() < 0.5 else {"note": "customer checkout"}
    variants = [
        {"order_ref": order_id},
        {"merchant_order_id": order_id, "source": "web"},
        {"ref": order_id.replace("ORD", "")},
        {"internal_note": f"order {order_id}"},
    ]
    return random.choice(variants)


def generate_narration_llm(prompt: str) -> str:
    """Optional real-LLM hook -- unused by default (USE_LLM=False)."""
    import os
    try:
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40,
            temperature=0.9,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None
