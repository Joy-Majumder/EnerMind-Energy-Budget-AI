# EnerMind — Adaptive AI for Personalized Energy Budgets

An adaptive AI system that forecasts household energy consumption, allocates personalized energy budgets for household members, and issues actionable recommendations and alerts to reduce energy use.

✓ **Complete implementation** — Ready-to-use prototype with ML pipeline  
✓ **Comprehensive documentation** — Validation scripts, reports, and paper summaries  
✓ **Multiple dataset options** — UCI ML datasets, sample data, or custom CSVs  
✓ **Easy to extend** — Modular design for custom forecasters and recommendation engines  

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Features

- LSTM-based forecasting (128 + 64 units) with Random Forest fallback
- Personalized budget allocation per household member
- Monthly adaptive adjustment based on actual consumption
- Multi-level alerts (warning at 80%, critical at 95%)
- Smart recommendations with appliance-level insights

## What's Included

**Core:** `main.py`, `system.py`, `models.py`, `core.py`, `preprocessing.py`, `utils.py`  
**Validation:** `train_and_validate.py`, `test_all_datasets.py`  
**Data:** `DataSets/` (UCI ML samples)  
**Reports:** `RESULTS_VALIDATION.md`, `COMPLETE_VALIDATION_SUMMARY.md`, `PAPER_UPDATES_SUMMARY.md`  

## Datasets

- UCI ID 235: Household Power Consumption (35K+ 15-min readings)
- UCI ID 50379: Electricity Load Diagrams (175K+ multi-household readings)
- UCI ID 242: Energy Efficiency (768 records)

Add custom CSV with columns: `timestamp`, `consumption_kwh`

## For Paper Submission

Run `python test_all_datasets.py`, then copy numbers from `PAPER_UPDATES_SUMMARY.md` into Table I/II.

## Configuration

Edit `config.py` for budget thresholds, alert levels, LSTM hyperparameters, and member profiles.

## Contact

Joy G. Majumdar — jmajumdar2520249@bscse.uiu.ac.bd

---

**Status:** Production-ready prototype
