"""
data_processing.py
------------------
Generates a synthetic PaySim-like dataset and splits it into two
simulated systems (internal_ledger.csv and bank_statement.csv)
with realistic discrepancies injected.
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(
    filename="logs/finops.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1.  SYNTHETIC DATA GENERATOR (no CSV needed)
# ─────────────────────────────────────────────
def generate_paysim_like(n_rows: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """
    Creates a PaySim-style DataFrame when the real CSV is absent.
    Columns mirror the original PaySim schema.
    """
    rng = np.random.default_rng(seed)

    tx_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
    weights   = [0.35, 0.25, 0.20, 0.15, 0.05]

    senders   = [f"C{rng.integers(1_000_000, 9_999_999)}" for _ in range(n_rows)]
    receivers = [f"M{rng.integers(1_000_000, 9_999_999)}" if t in ("PAYMENT","DEBIT")
                 else f"C{rng.integers(1_000_000, 9_999_999)}"
                 for t in rng.choice(tx_types, n_rows, p=weights)]

    amounts = np.round(rng.exponential(scale=50_000, size=n_rows), 2)
    amounts = np.clip(amounts, 10, 10_000_000)

    old_bal_orig  = np.round(rng.uniform(0, 500_000, n_rows), 2)
    new_bal_orig  = np.round(np.maximum(old_bal_orig - amounts, 0), 2)
    old_bal_dest  = np.round(rng.uniform(0, 500_000, n_rows), 2)
    new_bal_dest  = np.round(old_bal_dest + amounts, 2)

    is_fraud = (amounts > 800_000).astype(int)
    is_flagged_fraud = is_fraud.copy()

    df = pd.DataFrame({
        "step":          rng.integers(1, 744, n_rows),
        "type":          rng.choice(tx_types, n_rows, p=weights),
        "amount":        amounts,
        "nameOrig":      senders,
        "oldbalanceOrg": old_bal_orig,
        "newbalanceOrig":new_bal_orig,
        "nameDest":      receivers,
        "oldbalanceDest":old_bal_dest,
        "newbalanceDest":new_bal_dest,
        "isFraud":       is_fraud,
        "isFlaggedFraud":is_flagged_fraud,
        "budget_amount": amounts * rng.uniform(0.9, 1.1, n_rows),
    })
    df.index.name = "transaction_id"
    df = df.reset_index()
    df["transaction_id"] = ["T" + str(i).zfill(7) for i in df["transaction_id"]]
    logger.info(f"Generated synthetic PaySim dataset with {n_rows} rows.")
    return df


# ─────────────────────────────────────────────
# 2.  LOAD  (real CSV  OR  synthetic)
# ─────────────────────────────────────────────
def load_data(filepath: str = "data/PS_20174392719_1491204439457_log.csv",
              n_rows: int = 100_000) -> pd.DataFrame:
    """
    Loads the first n_rows from the PaySim CSV.
    Falls back to synthetic generation when the file is missing.
    """
    if os.path.exists(filepath):
        logger.info(f"Loading dataset from {filepath}")
        df = pd.read_csv(filepath, nrows=n_rows)
        df = df.rename(columns={"step": "step"})
        df.index.name = "row_id"
        df = df.reset_index()
        df["transaction_id"] = ["T" + str(i).zfill(7) for i in df["row_id"]]
        df.drop(columns=["row_id"], inplace=True)
    else:
        logger.warning("PaySim CSV not found – generating synthetic data.")
        print("⚠️  PaySim CSV not found. Generating synthetic dataset …")
        df = generate_paysim_like(n_rows)

    # Basic type safety
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    
    # Ensure budget_amount exists for variance analysis
    if "budget_amount" not in df.columns:
        rng = np.random.default_rng(42)
        df["budget_amount"] = df["amount"] * rng.uniform(0.9, 1.1, len(df))
        
    logger.info(f"Dataset loaded: {len(df):,} rows, {df.shape[1]} columns.")
    return df


# ─────────────────────────────────────────────
# 3.  DISCREPANCY INJECTOR
# ─────────────────────────────────────────────
def inject_discrepancies(df: pd.DataFrame, seed: int = 99) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits df into internal_ledger and bank_statement, then injects:
      - Missing transactions in bank (2 %)
      - Missing transactions in ledger (1 %)
      - Amount mismatches (1.5 %)
      - Duplicate transactions in bank (0.5 %)
      - Vendor/name inconsistencies (1 %)
    Returns (ledger_df, bank_df)
    """
    rng = np.random.default_rng(seed)
    n   = len(df)

    # ---- build ledger (near-clean copy) ---------------------------------
    ledger = df[[
        "transaction_id", "type", "amount", "budget_amount",
        "nameOrig", "nameDest",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "isFraud"
    ]].copy()
    ledger.columns = [
        "transaction_id", "transaction_type", "amount", "budget_amount",
        "sender", "receiver",
        "sender_old_balance", "sender_new_balance",
        "receiver_old_balance", "receiver_new_balance",
        "is_fraud"
    ]
    # Create timestamps across 3 months for MoM variance
    dates = pd.date_range("2023-01-01", periods=n, freq="1min")
    ledger["date"] = dates.strftime("%Y-%m-%d")
    ledger["month"] = dates.strftime("%B")
    ledger["currency"] = "INR"

    # ---- bank starts as a copy of ledger --------------------------------
    bank = ledger.copy()

    # ── a) Missing in bank (drop ~2 % rows) ──────────────────────────────
    missing_in_bank = rng.choice(n, size=int(n * 0.02), replace=False)
    bank = bank.drop(index=missing_in_bank).reset_index(drop=True)
    logger.info(f"Injected {len(missing_in_bank)} missing-in-bank rows.")

    # ── b) Missing in ledger (drop ~1 % rows from ledger) ────────────────
    remaining_idx = list(range(n))
    still_in_bank = set(bank["transaction_id"])
    ledger_candidates = [i for i in remaining_idx if ledger.iloc[i]["transaction_id"] in still_in_bank]
    missing_in_ledger = rng.choice(ledger_candidates,
                                   size=int(n * 0.01), replace=False)
    ledger = ledger.drop(index=missing_in_ledger).reset_index(drop=True)
    logger.info(f"Injected {len(missing_in_ledger)} missing-in-ledger rows.")

    # ── c) Amount mismatches in bank (~1.5 %) ─────────────────────────────
    mismatch_mask = rng.random(len(bank)) < 0.015
    noise = rng.uniform(0.01, 500, mismatch_mask.sum())
    bank.loc[mismatch_mask, "amount"] = (
        bank.loc[mismatch_mask, "amount"] + noise
    ).round(2)
    logger.info(f"Injected {mismatch_mask.sum()} amount-mismatch rows.")

    # ── d) Duplicate transactions in bank (~0.5 %) ───────────────────────
    dup_idx  = rng.choice(len(bank), size=int(len(bank) * 0.005), replace=False)
    dup_rows = bank.iloc[dup_idx].copy()
    bank = pd.concat([bank, dup_rows], ignore_index=True)
    logger.info(f"Injected {len(dup_idx)} duplicate rows into bank.")

    # ── e) Vendor/name inconsistencies in bank (~1 %) ────────────────────
    name_mask = rng.random(len(bank)) < 0.01
    bank.loc[name_mask, "sender"] = bank.loc[name_mask, "sender"].str.replace(
        r"C(\d+)", lambda m: "c" + m.group(1), regex=True   # lowercase prefix
    )
    logger.info(f"Injected {name_mask.sum()} name-inconsistency rows.")

    # ── Save to CSV ────────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    ledger.to_csv("data/internal_ledger.csv", index=False)
    bank.to_csv("data/bank_statement.csv",    index=False)
    logger.info("Saved internal_ledger.csv and bank_statement.csv.")
    print(f"✅ internal_ledger.csv  →  {len(ledger):,} rows")
    print(f"✅ bank_statement.csv   →  {len(bank):,} rows")

    return ledger, bank


# ─────────────────────────────────────────────
# 4.  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived numeric features useful for ML."""
    df = df.copy()
    df["balance_diff_sender"]   = df["sender_old_balance"]   - df["sender_new_balance"]
    df["balance_diff_receiver"] = df["receiver_new_balance"] - df["receiver_old_balance"]
    df["amount_to_sender_bal"]  = df["amount"] / (df["sender_old_balance"] + 1)
    df["is_round_amount"]       = (df["amount"] % 1000 == 0).astype(int)

    type_map = {t: i for i, t in enumerate(df["transaction_type"].unique())}
    df["type_encoded"] = df["transaction_type"].map(type_map).fillna(-1).astype(int)
    return df


# ─────────────────────────────────────────────
# 5.  QUICK SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    raw = load_data(n_rows=50_000)
    ledger, bank = inject_discrepancies(raw)
    ledger_fe = engineer_features(ledger)
    print(f"\nFeature-engineered ledger shape: {ledger_fe.shape}")
    print(ledger_fe.head(3).to_string())
