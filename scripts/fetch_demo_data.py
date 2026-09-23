"""Fetch demo data for the notebook + web visualisation.

Downloads Banking77 (13k real bank customer queries, CC-BY-4.0) from its
upstream GitHub mirror and maps its 77 fine intents to 7 coarse groups that
fit Laya's sweet spot (<=20 options; see TUTORIAL.md).

Outputs (committed, so the notebook runs offline):
    data/banking77_test.csv    text,fine_label,coarse_label (3,080 rows)
    data/banking77_train.csv   text,fine_label,coarse_label (10,003 rows)
    data/coarse_labels.json   {coarse_label: description}  (also the Laya criteria)
    data/DATA_README.md       source, license, mapping

Stdlib only. Idempotent: skips download when outputs exist unless --force.

    uv run scripts/fetch_demo_data.py [--force]
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# 77 fine intents -> 7 coarse groups. Every fine label must appear exactly once;
# the script asserts full coverage against categories.json so upstream drift fails loudly.
COARSE: dict[str, str] = {
    "cards": "physical or virtual cards: ordering, delivery, activation, PIN, loss, theft, card not working",
    "transfers": "sending or receiving money: transfers, direct debits, beneficiaries, transfer status and timing",
    "topup": "adding money to the account: top-ups by card, cash or bank transfer, limits and failures",
    "fees_charges": "unexpected fees or wrong amounts: extra charges, wrong exchange rates, double charges",
    "refunds_disputes": "refunds and problem payments: refund requests, declined or unrecognised payments",
    "account_identity": "account settings and identity: personal details, verification, closing the account",
    "help_info": "general information: exchange rates, balances, supported countries, cards and currencies",
}

FINE_TO_COARSE: dict[str, str] = {
    # --- cards (22) ---
    "card_arrival": "cards", "card_linking": "cards", "card_delivery_estimate": "cards",
    "card_not_working": "cards", "lost_or_stolen_card": "cards", "pin_blocked": "cards",
    "contactless_not_working": "cards", "compromised_card": "cards",
    "passcode_forgotten": "cards", "change_pin": "cards", "card_swallowed": "cards",
    "activate_my_card": "cards", "card_about_to_expire": "cards",
    "apple_pay_or_google_pay": "cards", "get_physical_card": "cards",
    "getting_virtual_card": "cards", "getting_spare_card": "cards",
    "order_physical_card": "cards", "get_disposable_virtual_card": "cards",
    "disposable_card_limits": "cards", "virtual_card_not_working": "cards",
    "card_acceptance": "cards",
    # --- transfers (10) ---
    "cancel_transfer": "transfers", "transfer_not_received_by_recipient": "transfers",
    "declined_transfer": "transfers", "pending_transfer": "transfers",
    "failed_transfer": "transfers", "transfer_timing": "transfers",
    "transfer_into_account": "transfers", "receiving_money": "transfers",
    "direct_debit_payment_not_recognised": "transfers",
    "beneficiary_not_allowed": "transfers",
    # --- topup (10) ---
    "automatic_top_up": "topup", "pending_top_up": "topup", "top_up_limits": "topup",
    "top_up_reverted": "topup", "topping_up_by_card": "topup",
    "top_up_by_cash_or_cheque": "topup", "top_up_failed": "topup",
    "verify_top_up": "topup", "top_up_by_card_charge": "topup",
    "top_up_by_bank_transfer_charge": "topup",
    # --- fees_charges (9) ---
    "extra_charge_on_statement": "fees_charges", "card_payment_fee_charged": "fees_charges",
    "exchange_charge": "fees_charges", "cash_withdrawal_charge": "fees_charges",
    "transfer_fee_charged": "fees_charges", "transaction_charged_twice": "fees_charges",
    "card_payment_wrong_exchange_rate": "fees_charges",
    "wrong_exchange_rate_for_cash_withdrawal": "fees_charges",
    "wrong_amount_of_cash_received": "fees_charges",
    # --- refunds_disputes (9) ---
    "request_refund": "refunds_disputes", "Refund_not_showing_up": "refunds_disputes",
    "reverted_card_payment?": "refunds_disputes", "declined_card_payment": "refunds_disputes",
    "declined_cash_withdrawal": "refunds_disputes",
    "pending_card_payment": "refunds_disputes",
    "pending_cash_withdrawal": "refunds_disputes",
    "cash_withdrawal_not_recognised": "refunds_disputes",
    "card_payment_not_recognised": "refunds_disputes",
    # --- account_identity (8) ---
    "edit_personal_details": "account_identity", "why_verify_identity": "account_identity",
    "unable_to_verify_identity": "account_identity", "verify_my_identity": "account_identity",
    "verify_source_of_funds": "account_identity", "terminate_account": "account_identity",
    "age_limit": "account_identity", "lost_or_stolen_phone": "account_identity",
    # --- help_info (9) ---
    "exchange_rate": "help_info", "fiat_currency_support": "help_info",
    "exchange_via_app": "help_info", "supported_cards_and_currencies": "help_info",
    "visa_or_mastercard": "help_info", "country_support": "help_info",
    "atm_support": "help_info", "balance_not_updated_after_bank_transfer": "help_info",
    "balance_not_updated_after_cheque_or_cash_deposit": "help_info",
}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "jev-laya-test"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> None:
    force = "--force" in sys.argv
    DATA.mkdir(exist_ok=True)
    out_test, out_train, out_json = (
        DATA / "banking77_test.csv", DATA / "banking77_train.csv", DATA / "coarse_labels.json")
    if out_test.exists() and out_train.exists() and out_json.exists() and not force:
        print(f"exists, skipping (use --force): {out_test}")
        return

    categories = json.loads(get(f"{BASE}/categories.json"))
    assert len(categories) == 77, f"upstream changed: {len(categories)} intents"
    missing = [c for c in categories if c not in FINE_TO_COARSE]
    extra = [c for c in FINE_TO_COARSE if c not in categories]
    assert not missing and not extra, f"mapping drift: missing={missing} extra={extra}"

    def remap(split: str) -> list:
        raw = get(f"{BASE}/{split}.csv").decode("utf-8").splitlines()
        rows = list(csv.DictReader(raw))
        assert set(rows[0]) == {"text", "category"}, rows[0].keys()
        return [(r["text"], r["category"], FINE_TO_COARSE[r["category"]]) for r in rows]

    for path, split in ((out_test, "test"), (out_train, "train")):
        mapped = remap(split)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text", "fine_label", "coarse_label"])
            w.writerows(mapped)
        print(f"wrote {path} ({len(mapped)} rows)")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(COARSE, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {out_json} ({len(COARSE)} groups)")


if __name__ == "__main__":
    main()
