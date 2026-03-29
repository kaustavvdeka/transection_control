# 💰 FinOps Agent — Financial Reconciliation & Anomaly Detection

A complete end-to-end Python project that simulates a **Financial Operations Agent** capable of:

- 🔍 Reconciling two financial systems (internal ledger vs bank statement)
- ⚠️ Detecting discrepancies (missing txns, mismatches, duplicates, spikes)
- 🧠 Root-cause analysis with human-readable explanations
- 💸 Financial impact calculation in ₹ INR
- 🤖 ML-based anomaly detection (Isolation Forest)
- 📊 Interactive Streamlit dashboard

---

## 📁 Project Structure

```
finops_agent/
├── main.py                    ← Entry point (CLI runner)
├── data_processing.py         ← Data loading, synthetic generation, feature engineering
├── requirements.txt
├── agents/
│   └── finops_agent.py        ← Core FinOpsAgent class
├── models/
│   └── anomaly.py             ← Isolation Forest anomaly detector
├── dashboard/
│   └── app.py                 ← Streamlit UI
├── data/                      ← Auto-generated CSVs
│   ├── internal_ledger.csv
│   ├── bank_statement.csv
│   └── issues_report.csv
├── models/                    ← Saved ML model (after first run)
│   ├── isolation_forest.joblib
│   └── scaler.joblib
└── logs/
    └── finops.log             ← Agent decision log
```

---

## ⚙️ Installation

```bash
# 1. Clone / copy the project
cd finops_agent

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

### Option A — Full run (100k rows)
```bash
python main.py
```

### Option B — Fast demo (20k rows, no ML)
```bash
python main.py --rows 20000 --no-ml
```

### Option C — Custom tolerance
```bash
python main.py --rows 50000 --tol 1.0
```

### CLI flags
| Flag | Default | Description |
|------|---------|-------------|
| `--rows` | 100000 | Number of rows to process |
| `--no-ml` | off | Disable Isolation Forest |
| `--tol` | 0.50 | Amount mismatch tolerance (₹) |

---

## 📊 Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

Then open `http://localhost:8501` in your browser.

> **Tip:** Click **▶ Run Full Pipeline** in the sidebar to generate fresh data.

---

## 📂 Using Real PaySim Data (Optional)

1. Download from Kaggle: [PaySim Financial Dataset](https://www.kaggle.com/ntnu-testimon/paysim1)
2. Place the CSV at:
   ```
   data/PS_20174392719_1491204439457_log.csv
   ```
3. Run the pipeline normally — it auto-detects the real file.

Without the file, the system generates a **realistic synthetic dataset** automatically.

---

## 🔍 Sample Output

```
🚀  FinOps Agent starting …

✅ internal_ledger.csv  →  99,700 rows
✅ bank_statement.csv   →  99,547 rows

============================================================
  🔍  RECONCILIATION SUMMARY
============================================================
  Total Transactions :     99,700
  Matched            :     95,841
  Issues Found       :      4,629
------------------------------------------------------------
  ⚠️   DISCREPANCIES
------------------------------------------------------------
  Amount Mismatch               :   1,404  |  ₹    350,985.64
  Balance Inconsistency         :   9,512  |  ₹988,934,273.37
  Duplicate                     :     492  |  ₹  7,184,083.23
  Missing In Bank               :   2,002  |  ₹110,246,519.17
  Missing In Ledger             :     997  |  ₹          0.00
  Ml Anomaly                    :     843  |  ₹211,119,135.65
  Transaction Spike             :     281  |  ₹  2,871,956.85
------------------------------------------------------------
  💰  Total Variance  : ₹1,320,706,953.91
============================================================

  🔔  HIGH-IMPACT ALERTS
  🚨 Flag transaction T0008684 | missing_in_bank | ₹158,534
  🚨 Flag transaction T0026465 | amount_mismatch | ₹87,887
  ...
```

---

## 🧪 Testing Individual Modules

```bash
# Test data generation
python data_processing.py

# Test anomaly model
python models/anomaly.py
```

---

## 🏗️ Architecture

```
main.py
  └─► FinOpsAgent.run()
        ├─ load_data()          → data_processing.py (load + inject discrepancies)
        ├─ reconcile()          → exact ID match + composite-key fuzzy fallback
        ├─ detect_issues()      → balance check + ML anomaly + spike detection
        ├─ analyze_root_cause() → maps status → human explanation
        ├─ calculate_impact()   → ₹ variance per category
        ├─ _generate_alerts()   → console alerts for high-value transactions
        └─ save_issues()        → data/issues_report.csv
```

---

## 📦 Key Libraries

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, manipulation, reconciliation |
| `scikit-learn` | Isolation Forest anomaly detection |
| `streamlit` | Interactive dashboard |
| `matplotlib` | Charts in dashboard |
| `joblib` | Model persistence |

---

## 💡 Hackathon Tips

- **Speed**: Use `--rows 20000` for quick demos
- **Real data**: Drop PaySim CSV into `data/` folder
- **Extend**: Add email alerts, webhook triggers, or database connectors in `agents/finops_agent.py`
- **Tune ML**: Adjust `contamination` param in `AnomalyDetector(contamination=0.02)`

---

*Built with ❤️ for FinOps hackathons | Python 3.10+*
