"""
agents/finops_agent.py
----------------------
Core FinOps Agent that orchestrates:
  • Data loading
  • Reconciliation
  • Discrepancy detection
  • Root-cause analysis
  • Financial impact calculation
  • ML anomaly detection
  • Alert generation & report export
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data_processing import load_data, inject_discrepancies, engineer_features
from models.anomaly   import AnomalyDetector
from agents.genai_agent import GenAIAgent

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Fuzzy name matching (pure Python, no external lib required)
# ─────────────────────────────────────────────────────────────────────────────
def _simple_similarity(a: str, b: str) -> float:
    """
    Jaccard similarity on character bigrams.
    Returns 0.0–1.0.  Fast enough for our needs.
    """
    if not a or not b:
        return 0.0
    a, b = str(a).lower(), str(b).lower()
    if a == b:
        return 1.0
    bigrams_a = set(a[i:i+2] for i in range(len(a)-1))
    bigrams_b = set(b[i:i+2] for i in range(len(b)-1))
    if not bigrams_a or not bigrams_b:
        return 0.0
    return len(bigrams_a & bigrams_b) / len(bigrams_a | bigrams_b)


def _fuzzy_name_match(s1: pd.Series, s2: pd.Series, threshold: float = 0.8) -> pd.Series:
    """Vectorised fuzzy comparison; returns boolean Series."""
    return pd.Series(
        [_simple_similarity(a, b) >= threshold for a, b in zip(s1, s2)],
        index=s1.index
    )


# ─────────────────────────────────────────────────────────────────────────────
#  FinOps Agent
# ─────────────────────────────────────────────────────────────────────────────
class FinOpsAgent:
    """
    End-to-end Financial Operations Agent.

    Parameters
    ----------
    n_rows          : rows to load from source data
    amount_tol      : tolerance (INR) for amount-mismatch detection
    use_ml          : enable Isolation Forest anomaly detection
    spike_z_thresh  : Z-score threshold for transaction-volume spikes
    """

    def __init__(
        self,
        n_rows:         int   = 100_000,
        amount_tol:     float = 0.50,
        use_ml:         bool  = True,
        spike_z_thresh: float = 3.0,
    ):
        self.n_rows         = n_rows
        self.amount_tol     = amount_tol
        self.use_ml         = use_ml
        self.spike_z_thresh = spike_z_thresh

        # Data holders
        self.ledger:   pd.DataFrame | None = None
        self.bank:     pd.DataFrame | None = None
        self.ledger_fe:pd.DataFrame | None = None

        # Result holders
        self.reconciliation_df: pd.DataFrame | None = None
        self.discrepancies:     pd.DataFrame | None = None
        self.issues_df:         pd.DataFrame | None = None
        self.impact_summary:    dict                = {}

        # ML & GenAI
        self.detector = AnomalyDetector(contamination=0.02)
        self.genai    = GenAIAgent()
        self._ml_trained = False
        
        # GenAI Insights
        self.closing_narrative: str | None = None

    # =========================================================================
    #  1.  LOAD DATA
    # =========================================================================
    def load_data(self):
        """Load raw PaySim data and create ledger + bank files."""
        logger.info("=== FinOpsAgent: load_data() ===")
        raw = load_data(n_rows=self.n_rows)
        self.ledger, self.bank = inject_discrepancies(raw)
        self.ledger_fe = engineer_features(self.ledger)
        logger.info(f"Ledger: {len(self.ledger):,} rows | Bank: {len(self.bank):,} rows")

    # =========================================================================
    #  2.  RECONCILIATION
    # =========================================================================
    def reconcile(self) -> pd.DataFrame:
        """
        Match transactions between ledger and bank.

        Strategy
        --------
        Primary key : transaction_id
        Fallback key: (amount_rounded, sender_norm, receiver_norm)

        Returns a reconciliation DataFrame with a 'status' column.
        """
        logger.info("=== FinOpsAgent: reconcile() ===")
        ledger = self.ledger.copy()
        bank   = self.bank.copy()

        # Normalise for fuzzy key
        for df in (ledger, bank):
            df["_sender_norm"]   = df["sender"].str.upper().str.strip()
            df["_receiver_norm"] = df["receiver"].str.upper().str.strip()
            df["_amount_key"]    = df["amount"].round(0)
            df["_composite_key"] = (
                df["_amount_key"].astype(str) + "|" +
                df["_sender_norm"]            + "|" +
                df["_receiver_norm"]
            )

        # ── a) Exact ID match ────────────────────────────────────────────
        ledger_ids = set(ledger["transaction_id"])
        bank_ids   = set(bank["transaction_id"])

        matched_ids      = ledger_ids & bank_ids
        only_in_ledger   = ledger_ids - bank_ids
        only_in_bank     = bank_ids   - ledger_ids

        # ── b) For missing IDs try composite-key match ────────────────────
        unmatched_ledger_df = ledger[ledger["transaction_id"].isin(only_in_ledger)]
        unmatched_bank_df   = bank  [bank  ["transaction_id"].isin(only_in_bank  )]

        composite_ledger = set(unmatched_ledger_df["_composite_key"])
        composite_bank   = set(unmatched_bank_df  ["_composite_key"])
        composite_matched = composite_ledger & composite_bank

        # Reclassify: if composite match exists → "matched (fuzzy)"
        ledger_fuzzy_ids = set(
            unmatched_ledger_df[
                unmatched_ledger_df["_composite_key"].isin(composite_matched)
            ]["transaction_id"]
        )
        bank_fuzzy_ids = set(
            unmatched_bank_df[
                unmatched_bank_df["_composite_key"].isin(composite_matched)
            ]["transaction_id"]
        )

        only_in_ledger_final = only_in_ledger - ledger_fuzzy_ids
        only_in_bank_final   = only_in_bank   - bank_fuzzy_ids

        # ── c) Detect duplicates in bank ─────────────────────────────────
        bank_dup_mask = bank.duplicated(subset=["transaction_id"], keep="first")
        duplicate_ids = set(bank[bank_dup_mask]["transaction_id"])

        # ── d) Amount mismatches in exact-ID matches ──────────────────────
        merged = ledger.merge(
            bank[["transaction_id", "amount"]],
            on="transaction_id", suffixes=("_ledger","_bank"), how="inner"
        )
        mismatch_mask = abs(merged["amount_ledger"] - merged["amount_bank"]) > self.amount_tol
        mismatch_ids  = set(merged[mismatch_mask]["transaction_id"])

        # ── e) Build result DataFrame ─────────────────────────────────────
        all_ids = ledger_ids | bank_ids
        records = []
        for tid in all_ids:
            if tid in mismatch_ids:
                status = "amount_mismatch"
            elif tid in duplicate_ids:
                status = "duplicate"
            elif tid in matched_ids and tid not in mismatch_ids:
                status = "matched"
            elif tid in ledger_fuzzy_ids or tid in bank_fuzzy_ids:
                status = "matched_fuzzy"
            elif tid in only_in_ledger_final:
                status = "missing_in_bank"
            elif tid in only_in_bank_final:
                status = "missing_in_ledger"
            else:
                status = "matched"
            records.append({"transaction_id": tid, "status": status})

        rec_df = pd.DataFrame(records)

        # Attach amounts for impact calculation
        rec_df = rec_df.merge(
            ledger[["transaction_id","amount","sender","receiver","transaction_type"]],
            on="transaction_id", how="left"
        )
        rec_df = rec_df.merge(
            bank[["transaction_id","amount"]].rename(columns={"amount":"bank_amount"}),
            on="transaction_id", how="left"
        )
        rec_df["amount_variance"] = (
            rec_df["amount"].fillna(0) - rec_df["bank_amount"].fillna(0)
        ).abs()

        self.reconciliation_df = rec_df
        logger.info(f"Reconciliation complete. Statuses:\n{rec_df['status'].value_counts().to_dict()}")
        return rec_df

    # =========================================================================
    #  3.  DETECT ISSUES
    # =========================================================================
    def detect_issues(self) -> pd.DataFrame:
        """
        Combines reconciliation flags + ML anomaly flags + spike detection.
        Returns a unified issues DataFrame.
        """
        logger.info("=== FinOpsAgent: detect_issues() ===")
        if self.reconciliation_df is None:
            self.reconcile()

        issues = self.reconciliation_df[
            self.reconciliation_df["status"] != "matched"
        ].copy()

        # ── Balance inconsistency check ───────────────────────────────────
        ledger = self.ledger.copy()
        ledger["balance_inconsistency"] = (
            abs(
                (ledger["sender_old_balance"] - ledger["amount"]) -
                 ledger["sender_new_balance"]
            ) > self.amount_tol
        )
        bad_bal_ids = set(ledger[ledger["balance_inconsistency"]]["transaction_id"])

        # Add balance-inconsistency rows not already in issues
        new_bal_issues = ledger[
            ledger["transaction_id"].isin(bad_bal_ids) &
            ~ledger["transaction_id"].isin(set(issues["transaction_id"]))
        ][["transaction_id","amount","sender","receiver","transaction_type"]].copy()
        new_bal_issues["status"]           = "balance_inconsistency"
        new_bal_issues["bank_amount"]      = np.nan
        new_bal_issues["amount_variance"]  = np.nan
        issues = pd.concat([issues, new_bal_issues], ignore_index=True)

        # ── ML anomaly detection ──────────────────────────────────────────
        if self.use_ml and self.ledger_fe is not None:
            try:
                if not self._ml_trained:
                    self.detector.train(self.ledger_fe)
                    self.detector.save()
                    self._ml_trained = True
                preds  = self.detector.predict(self.ledger_fe)
                scores = self.detector.anomaly_scores(self.ledger_fe)
                anomaly_ids = set(
                    self.ledger_fe.loc[preds == -1, "transaction_id"]
                )
                new_ml_ids = anomaly_ids - set(issues["transaction_id"])
                ml_rows = self.ledger[
                    self.ledger["transaction_id"].isin(new_ml_ids)
                ][["transaction_id","amount","sender","receiver","transaction_type"]].copy()
                ml_rows["status"]          = "ml_anomaly"
                ml_rows["bank_amount"]     = np.nan
                ml_rows["amount_variance"] = np.nan
                issues = pd.concat([issues, ml_rows], ignore_index=True)
                logger.info(f"ML flagged {len(anomaly_ids):,} anomalies.")
            except Exception as e:
                logger.warning(f"ML detection failed: {e}")

        # ── Transaction spike detection (per sender) ─────────────────────
        tx_counts = self.ledger.groupby("sender")["transaction_id"].count()
        mean_c, std_c = tx_counts.mean(), tx_counts.std()
        spike_senders = set(tx_counts[tx_counts > mean_c + self.spike_z_thresh * std_c].index)
        spike_rows = self.ledger[
            self.ledger["sender"].isin(spike_senders) &
            ~self.ledger["transaction_id"].isin(set(issues["transaction_id"]))
        ][["transaction_id","amount","sender","receiver","transaction_type"]].copy()
        spike_rows["status"]          = "transaction_spike"
        spike_rows["bank_amount"]     = np.nan
        spike_rows["amount_variance"] = np.nan
        issues = pd.concat([issues, spike_rows], ignore_index=True)

        self.issues_df = issues.drop_duplicates(subset=["transaction_id"]).reset_index(drop=True)
        logger.info(f"Total issues detected: {len(self.issues_df):,}")
        return self.issues_df

    # =========================================================================
    #  4.  ROOT CAUSE ANALYSIS
    # =========================================================================
    def analyze_root_cause(self) -> pd.DataFrame:
        """
        Appends a human-readable 'root_cause' column to issues_df.
        """
        logger.info("=== FinOpsAgent: analyze_root_cause() ===")
        if self.issues_df is None:
            self.detect_issues()

        cause_map = {
            "missing_in_bank":       "Transaction missing in bank – possible processing delay or bank error",
            "missing_in_ledger":     "Transaction missing in ledger – possible unrecorded bank entry",
            "amount_mismatch":       "Amount mismatch – possible billing error or data-entry discrepancy",
            "duplicate":             "Duplicate transaction detected – possible double-charge or sync error",
            "balance_inconsistency": "Balance inconsistency detected – sender balance doesn't reconcile with amount",
            "ml_anomaly":            "ML model flagged unusual pattern – review for potential fraud",
            "transaction_spike":     "Abnormal transaction spike detected – unusually high frequency from sender",
            "matched_fuzzy":         "Matched via fuzzy keys – minor name/ID inconsistency present",
        }

        self.issues_df["root_cause"] = self.issues_df["status"].map(cause_map).fillna(
            "Unknown discrepancy – manual review required"
        )
        
        # ── GenAI-driven granular analysis ──────────────────────────────
        # Analyze top 5 high-impact issues via GenAI
        logger.info("Executing GenAI granular analysis for high-impact items...")
        top_issues = self.issues_df.sort_values("amount", ascending=False).head(5)
        
        # Add a column for GenAI-specific insights if requested (demo mode)
        self.issues_df["genai_insight"] = "Analysis pending..."
        
        for idx, row in top_issues.iterrows():
            insight = self.genai.analyze_discrepancy(row.to_dict())
            self.issues_df.at[idx, "genai_insight"] = insight

        logger.info("Root-cause analysis complete.")
        return self.issues_df

    # =========================================================================
    #  5.  FINANCIAL IMPACT
    # =========================================================================
    def calculate_impact(self) -> dict:
        """
        Computes total and per-category financial variance in INR.
        """
        logger.info("=== FinOpsAgent: calculate_impact() ===")
        if self.issues_df is None:
            self.analyze_root_cause()

        df = self.issues_df.copy()

        # For amount_mismatch use exact variance; for others use full amount
        df["impact_inr"] = np.where(
            df["status"] == "amount_mismatch",
            df["amount_variance"].fillna(0),
            df["amount"].fillna(0)
        )

        total_variance = df["impact_inr"].sum()
        per_category   = df.groupby("status")["impact_inr"].agg(
            count="count", total_impact="sum"
        ).reset_index()

        self.impact_summary = {
            "total_transactions": len(self.ledger),
            "total_issues":       len(df),
            "total_variance_inr": round(total_variance, 2),
            "per_category":       per_category.to_dict(orient="records"),
        }
        
        # ── GenAI Closing Narrative ──────────────────────────────────────
        logger.info("Generating GenAI Closing Narrative...")
        top_3 = df.sort_values("amount", ascending=False).head(3)
        self.closing_narrative = self.genai.generate_closing_report(self.impact_summary, top_3)
        self.impact_summary["closing_narrative"] = self.closing_narrative

        logger.info(f"Total variance: ₹{total_variance:,.2f}")
        return self.impact_summary

    # =========================================================================
    #  6.  ALERTS & REPORT
    # =========================================================================
    def _generate_alerts(self):
        """Print console alerts for high-impact transactions."""
        if self.issues_df is None:
            return
        HIGH_IMPACT = 50_000
        high = self.issues_df[self.issues_df["amount"].fillna(0) > HIGH_IMPACT]
        for _, row in high.head(20).iterrows():
            print(f"  🚨 Flag transaction {row['transaction_id']} "
                  f"| {row['status']} | ₹{row['amount']:,.0f}")
            logger.warning(f"ALERT: {row['transaction_id']} | {row['status']} | ₹{row['amount']:,.0f}")

    def save_issues(self, path: str = "data/issues_report.csv"):
        """Save issues DataFrame to CSV."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.issues_df is not None:
            self.issues_df.to_csv(path, index=False)
            logger.info(f"Issues saved → {path}")
            print(f"  💾 Issues report saved → {path}")

    def _print_summary(self):
        """Pretty-print reconciliation + impact summary."""
        s   = self.impact_summary
        rec = self.reconciliation_df["status"].value_counts().to_dict() if self.reconciliation_df is not None else {}

        print("\n" + "="*60)
        print("  🔍  RECONCILIATION SUMMARY")
        print("="*60)
        print(f"  Total Transactions : {s.get('total_transactions', 0):>10,}")
        print(f"  Matched            : {rec.get('matched', 0):>10,}")
        print(f"  Fuzzy Matched      : {rec.get('matched_fuzzy', 0):>10,}")
        print(f"  Issues Found       : {s.get('total_issues', 0):>10,}")
        print("-"*60)
        print("  ⚠️   DISCREPANCIES")
        print("-"*60)
        for cat in s.get("per_category", []):
            label = cat["status"].replace("_", " ").title()
            print(f"  {label:<30}: {cat['count']:>6,}  |  ₹{cat['total_impact']:>14,.2f}")
        print("-"*60)
        print(f"  💰  Total Variance  : ₹{s.get('total_variance_inr', 0):>14,.2f}")
        print("="*60)
        print()

    # =========================================================================
    #  7.  FULL RUN
    # =========================================================================
    def run(self):
        """Execute the complete FinOps pipeline."""
        print("\n🚀  FinOps Agent starting …\n")
        start = datetime.now()

        self.load_data()
        self.reconcile()
        self.detect_issues()
        self.analyze_root_cause()
        self.calculate_impact()

        self._print_summary()
        print("  🔔  HIGH-IMPACT ALERTS (top 20)")
        self._generate_alerts()
        self.save_issues()

        elapsed = (datetime.now() - start).total_seconds()
        print(f"\n✅  Pipeline complete in {elapsed:.1f}s")
        logger.info(f"Pipeline complete in {elapsed:.1f}s")

        return {
            "reconciliation": self.reconciliation_df,
            "issues":         self.issues_df,
            "impact":         self.impact_summary,
        }

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    # Run agent
    agent = FinOpsAgent(n_rows=20_000)
    agent.run()
