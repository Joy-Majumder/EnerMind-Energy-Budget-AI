# EnerMind — Adaptive AI for Personalized Energy Budgets

EnerMind is an adaptive AI system that forecasts household energy consumption,
allocates personalized energy budgets for household members, and issues
actionable recommendations and alerts to reduce energy use. This repository
contains the prototype implementation, validation scripts, and reproducible
results used to produce the companion paper.

Key features
- LSTM-based forecasting (128 + 64 units) with a Random Forest fallback
- Personalized budget allocation and monthly adjustment
- Real-time alerting and recommendation engine
- Reproducible training and multi-dataset validation scripts

Getting started
1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install core dependencies (adjust versions in `requirements.txt` if present):

```bash
pip install -r requirements.txt
```

3. Run the single-dataset validation (quick):

```bash
python train_and_validate.py
```

4. Run multi-dataset validation:

```bash
python test_all_datasets.py
```

What’s included
- `main.py` — demo orchestrator and system entry point
- `system.py`, `models.py`, `preprocessing.py` — core implementation
- `train_and_validate.py` — reproducible single-dataset training & validation
- `test_all_datasets.py` — runs validation across all CSVs in `DataSets/`
- `DataSets/` — sample datasets (may be large; include in repo only if you intend to publish raw data)
- `RESULTS_VALIDATION.md`, `COMPLETE_VALIDATION_SUMMARY.md`, `PAPER_UPDATES_SUMMARY.md` — generated human-readable reports
- `VALIDATION_RESULTS.json`, `MULTI_DATASET_VALIDATION.json` — machine-readable results

Notes & recommendations
- `DataSets/` contains sample CSVs used for validation; these files can be
  large and are commonly excluded from the main repo. If you want to publish
  raw datasets, prefer a separate data release or attach them as supplementary
  materials.
- The LSTM implementation can be enabled or disabled depending on environment
  (TensorFlow may not be available in every environment). The repo includes
  tree-based fallbacks for rapid validation and CI-friendly runs.

Reproducibility
- All validation scripts produce JSON outputs (`VALIDATION_RESULTS.json`,
  `MULTI_DATASET_VALIDATION.json`) and Markdown reports.
- To reproduce results, run `test_all_datasets.py` and inspect the generated
  JSON and Markdown files.

Which generated files to keep in the repo
- Keep: `RESULTS_VALIDATION.md`, `COMPLETE_VALIDATION_SUMMARY.md`, `PAPER_UPDATES_SUMMARY.md` — these explain runs and are useful for reviewers.
- Optional: `VALIDATION_RESULTS.json`, `MULTI_DATASET_VALIDATION.json` — keep for reproducibility or move to release assets if you prefer a smaller repository.
- Consider removing raw `DataSets/` from the main branch if licensing or size is a concern; instead provide scripts to download or regenerate samples.

Guidance for updating the paper
- Use `PAPER_UPDATES_SUMMARY.md` and `MASTER_SUMMARY_TABLE.md` for numbers to paste into Table I/II after you re-run final experiments in the target environment (GPU/TensorFlow for LSTM, system OpenMP for XGBoost).
- Note: the repository contains concrete RF and LR runs; XGBoost/TensorFlow results may be estimated if those packages are unavailable in your environment.

Contact
- Joy G. Majumdar — jmajumdar2520249@bscse.uiu.ac.bd

---

If you want, I can now: (A) remove `DataSets/` from the repo and add download instructions, (B) move large JSON outputs to a GitHub Release and keep the MD summaries, or (C) keep everything in the repo for maximum reproducibility. Tell me which option you prefer and I will apply it and push the change.

- ✓ Complete implementation
- ✓ Comprehensive documentation
- ✓ Multiple dataset options
- ✓ Easy to extend

**Start here:** `python main.py` 🚀

---

*Based on research by Joy Gopal Majumdar and Sneha Hazra*
>>>>>>> 3b59d3f (chore: add core EnerMind code, validation results and docs)
