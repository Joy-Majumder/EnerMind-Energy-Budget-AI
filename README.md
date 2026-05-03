# EnerMind: An Adaptive AI System for Personalized Energy Budget Planning

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

EnerMind is a Python prototype for personalized household energy budget planning. It uses an LSTM model to forecast end-of-month electricity consumption per household member and triggers alerts when usage approaches the allocated budget.

---

## Features

- Per-member energy budget allocation based on historical consumption
- LSTM-based monthly consumption forecasting
- Random Forest fallback for new users (cold-start)
- Alert thresholds: warning at 80%, critical at 95%, and projected overrun
- Appliance-level saving recommendations

---

## Project Structure

```
EnerMind/
├── data/                   # Raw and processed meter data (CSV)
├── src/
│   ├── preprocess.py       # Data cleaning and feature engineering
│   ├── train_lstm.py       # LSTM model training
│   ├── train_rf.py         # Random Forest fallback training
│   ├── budget_engine.py    # Budget allocation and adaptive updates
│   ├── alert_service.py    # Alert evaluation and recommendations
│   └── enermind.py         # Main entry point
├── models/                 # Saved model files
├── notebooks/              # Jupyter notebooks for exploration
├── config/
│   └── settings.yaml       # Configuration (thresholds, paths, parameters)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Joy-Majumder/EnerMind-An-Adaptive-AI-System-for-Personalized-Energy-Budget-Planning.git
cd EnerMind-An-Adaptive-AI-System-for-Personalized-Energy-Budget-Planning
pip install -r requirements.txt
```

### Dependencies

```
tensorflow==2.8
scikit-learn==1.0
pandas
numpy
```

---

## Usage

```bash
# 1. Preprocess raw meter data
python src/preprocess.py --data data/pecan_street.csv

# 2. Train the LSTM forecasting model
python src/train_lstm.py --epochs 200 --patience 20

# 3. Run the budget engine and alert service
python src/enermind.py --config config/settings.yaml
```

---

## Dataset

This prototype was developed and tested using the [Pecan Street Dataport](https://dataport.pecanstreet.org) dataset — 15-minute interval sub-metered energy data for residential households.

---

## Authors

- **Joy Gopal Majumdar** — United International University, Dhaka (jmajumdar2520249@bscse.uiu.ac.bd)
- **Sneha Hazra** — Daffodil International University, Dhaka

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.