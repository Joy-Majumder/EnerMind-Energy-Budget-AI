# EnerMind — Adaptive AI System for Personalized Energy Budget Planning

An AI-driven system for personalized, member-level household energy budgeting with real-time alerts and demand-response recommendations.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main prototype demo
python main.py
```

## Evaluation Suite

The evaluation suite addresses all reviewer feedback and produces reproducible results for the paper.

### Run All Evaluations

```bash
python run_all_evaluations.py
```

This executes all 5 evaluation components in sequence and consolidates findings into [`RESULTS.md`](RESULTS.md) and [`FINAL_EVALUATION_RESULTS.json`](FINAL_EVALUATION_RESULTS.json).

### Documentation & Reviewer Rebuttal
- **[`FINAL_REPORT.md`](FINAL_REPORT.md)** — Master publication-grade experimental report and synthesis.
- **[`RESULTS.md`](RESULTS.md)** — Consolidated experimental evaluation tables, sensitivity sweeps, cross-validation metrics, and robustness benchmarks.
- **[`DRAFT.md`](DRAFT.md)** — Point-by-point reviewer rebuttal and paper revision guide addressing all 6 major review comments.

### Individual Components

| Script | Reviewer Concern | Output |
|--------|-----------------|--------|
| `validate_all_models.py` | R1#6, R2#1: Model comparison needs real results | `REAL_MODEL_RESULTS.json` |
| `sensitivity_analysis.py` | R1#3, R2#3: Budget parameter k not justified | `SENSITIVITY_ANALYSIS_RESULTS.json` |
| `cross_validation.py` | R1#6, R2#4: Need statistical model comparison | `CROSS_VALIDATION_RESULTS.json` |
| `pipeline_evaluation.py` | R1#4, R2#4: Need full pipeline evaluation | `PIPELINE_EVALUATION_RESULTS.json` |
| `robustness_testing.py` | R1#5, R2#5: Need robustness testing | `ROBUSTNESS_TESTING_RESULTS.json` |

### What Each Component Does

1. **`validate_all_models.py`** — Trains all 5 architectures (MLP, LSTM, XGBoost, RF, LR) on real UCI-235 data. Produces updated Table I with actual (not estimated) numbers.

2. **`sensitivity_analysis.py`** — Sweeps comfort factor k ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0} and computes both theoretical Gaussian P(exceed) and empirical overrun rates. Produces data for the sensitivity analysis table.

3. **`cross_validation.py`** — Rolling-origin (walk-forward) cross-validation with 5 chronological folds. Reports mean ± std of MAE/RMSE. Runs Wilcoxon signed-rank paired test between top-2 models with effect size.

4. **`pipeline_evaluation.py`** — Replays historical daily traces through the complete budget → alert → recommendation pipeline. Reports alert precision, recall, false alarm rate, cool-down suppression, and recommendation coverage.

5. **`robustness_testing.py`** — Tests missing-data degradation (5/10/20/30% gaps), seasonal splits (train on H1 test on H2 and vice versa, plus per-quarter leave-one-out), and occupancy-change simulation (±30% level shift).

## Project Structure

```
EnerMind/
├── main.py                    # Main demo entry point
├── system.py                  # System orchestration
├── core.py                    # Budget, Alert, Recommendation engines
├── models.py                  # LSTM, RF, Hybrid forecasters
├── preprocessing.py           # Data cleaning & feature engineering
├── config.py                  # System configuration
├── data_generator.py          # Synthetic data for demo
├── utils.py                   # Utilities
├── validate_all_models.py     # [NEW] Real model training (all 5)
├── sensitivity_analysis.py    # [NEW] k-sensitivity sweep
├── cross_validation.py        # [NEW] Walk-forward CV + Wilcoxon
├── pipeline_evaluation.py     # [NEW] Full pipeline replay
├── robustness_testing.py      # [NEW] Missing data + seasonal tests
├── run_all_evaluations.py     # [NEW] Master runner
├── requirements.txt
└── DataSets/
    ├── uci_id_235_*.csv       # UCI-235 household data
    └── uci_id_50379_*.csv     # UCI-321 multi-household data
```

## Datasets

- **UCI 235** — Individual Household Electric Power Consumption (Hebrail & Berard, 2006)
- **UCI 321** — ElectricityLoadDiagrams20112014 (Trindade, 2015)
- **UCI 242** — Energy Efficiency (Tsanas & Xifara, 2012) — cross-sectional, excluded from primary analysis

## Citation

```bibtex
@misc{enermind2026,
  author = {Majumdar, Joy Gopal},
  title = {EnerMind: Prototype Source Code and Validation Scripts},
  year = {2026},
  url = {https://github.com/Joy-Majumder/EnerMind-Energy-Budget-AI}
}
```

## License

See repository for license details.
