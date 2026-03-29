"""
main.py
-------
Entry point for the FinOps Agent pipeline.

Usage
-----
    python main.py                    # full run (100 k rows)
    python main.py --rows 20000       # faster demo run
    python main.py --rows 10000 --no-ml
"""

import os
import sys
import argparse
import logging

# ── Ensure project root is on path ───────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

os.makedirs("logs",   exist_ok=True)
os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    filename="logs/finops.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
logging.getLogger().addHandler(console)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="FinOps Agent – Financial Reconciliation Pipeline")
    p.add_argument("--rows",  type=int, default=100_000, help="Rows to process (default 100000)")
    p.add_argument("--no-ml", action="store_true",       help="Disable ML anomaly detection")
    p.add_argument("--tol",   type=float, default=0.5,   help="Amount tolerance in INR (default 0.50)")
    return p.parse_args()


def main():
    args = parse_args()

    from agents.finops_agent import FinOpsAgent

    agent = FinOpsAgent(
        n_rows     = args.rows,
        amount_tol = args.tol,
        use_ml     = not args.no_ml,
    )

    results = agent.run()

    # ── Extra reconciliation stats ────────────────────────────────────────
    rec = results["reconciliation"]
    if rec is not None:
        print("\n📋  Reconciliation Status Counts:")
        for status, cnt in rec["status"].value_counts().items():
            print(f"     {status:<30} {cnt:>8,}")

    print("\n🏁  Done.  Check data/issues_report.csv for full details.")
    print("    To launch dashboard:  streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()
