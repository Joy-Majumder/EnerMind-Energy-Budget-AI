# EnerMind: An Adaptive AI System for Personalized Energy Budget Planning

> **An intelligent, LSTM-powered residential energy management platform that monitors, predicts, and controls household electricity consumption at the individual member level.**

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

EnerMind addresses the growing challenge of residential electricity management by combining real-time smart meter data, machine learning forecasting, and a personalised budget allocation engine — all in one cohesive platform. Inspired by personal financial budgeting, EnerMind assigns individual **energy quotas** to each household member based on their historical consumption patterns and adapts those quotas every billing cycle.

**Key highlights from experimental evaluation (Pecan Street Dataset):**

| Metric | Value |
|---|---|
| Forecasting accuracy (LSTM) | **94.3%** |
| MAE | **0.42 kWh** |
| Max energy reduction (Month 4) | **16.9%** |
| Annual saving equivalent | **~612 kWh / household** |

---

## ✨ Features

- **Real-time smart meter integration** — IEC 62056 (DLMS/COSEM) protocol, 15-minute interval ingestion via Apache Kafka → InfluxDB
- **LSTM-based consumption forecasting** — two stacked LSTM layers (128 → 64 units) with weekly retraining on rolling 6-month window
- **Cold-start handling** — hybrid Random Forest fallback for users with < 60 days of history, confidence-weighted blending
- **Personalised budget allocation** — per-member monthly quotas derived from historical mean ± comfort factor (k = 0.5), adaptive tightening each cycle (α = 0.10)
- **Intelligent alert & recommendation engine** — three-tier threshold system (warning 80%, critical 95%, projected overrun) with 4-hour cool-down and appliance-level saving suggestions
- **Multi-channel notifications** — push, email, and SMS with user-configurable preferences
- **Privacy-first design** — TLS 1.3 in transit, AES-256 at rest, optional edge processing; GDPR / Bangladesh PDPO compliant

---

## 🏗️ System Architecture

EnerMind is structured as a **layered, modular architecture** with four principal layers:

```
┌─────────────────────────────────────────────────────┐
│              User Interaction Layer                 │
│        (Dashboard · Alerts · Recommendations)       │
├─────────────────────────────────────────────────────┤
│            Budget Management Layer                  │
│   (Quota Allocation · Utilisation Ratio · Alerts)   │
├─────────────────────────────────────────────────────┤
│         Processing & Intelligence Layer             │
│     (LSTM Forecasting · Random Forest Fallback)     │
├─────────────────────────────────────────────────────┤
│             Data Acquisition Layer                  │
│   (Smart Meters · Kafka Broker · InfluxDB Store)    │
└─────────────────────────────────────────────────────┘
```

### Component Details

| Layer | Technology |
|---|---|
| Stream ingestion | Apache Kafka (per-household, per-member topics) |
| Time-series storage | InfluxDB |
| Primary forecasting model | LSTM (TensorFlow 2.8) |
| Cold-start fallback | Random Forest (scikit-learn 1.0) |
| Alert microservice | Event-driven, stateless evaluation function |
| Notification delivery | Multi-channel (push / email / SMS) |

---

## 📐 Methodology

### Budget Allocation

Each member's initial monthly energy budget **B_i** is calculated as:

```
B_i = C̄_i + k · σ_C_i
```

where `C̄_i` is mean daily consumption, `σ_C_i` is the standard deviation, and `k = 0.5` is the comfort factor. Budgets are updated each billing cycle with a learning rate **α = 0.10**.

### Consumption Forecasting

The projected end-of-month total **P_i** is:

```
P_i = A_i + R_i
```

where `A_i` is actual consumption so far and `R_i` is the LSTM-predicted remaining consumption. Budget utilisation **U_i = P_i / B_i** drives alert thresholds and dashboard indicators.

### LSTM Training Configuration

| Hyperparameter | Value |
|---|---|
| Architecture | 2-layer LSTM (128 → 64 units) + FC output |
| Optimiser | Adam (lr = 0.001) |
| Loss function | Mean Squared Error (MSE) |
| Batch size | 32 |
| Max epochs | 200 (early stopping, patience = 20) |
| Input window | 30-day sliding window |
| Retraining | Weekly on last 6 months |

---

## 📊 Experimental Results

### Model Comparison (Pecan Street Dataset, 50 households, 12 months)

| Model | MAE (kWh) | RMSE (kWh) | Accuracy (%) |
|---|---|---|---|
| **LSTM** | **0.42** | **0.61** | **94.3** |
| XGBoost | 0.49 | 0.68 | 93.1 |
| Random Forest | 0.58 | 0.79 | 91.2 |
| Linear Regression | 0.81 | 1.02 | 87.5 |

### Energy Reduction Over 4-Month Adoption Period (Variable Consumer Profile)

| Month | Before (kWh) | After (kWh) | Reduction (%) |
|---|---|---|---|
| Month 1 | 320 | 298 | 6.9% |
| Month 2 | 315 | 281 | 10.8% |
| Month 3 | 309 | 264 | 14.6% |
| Month 4 | 302 | 251 | **16.9%** |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Apache Kafka (for live stream ingestion)
- InfluxDB (for time-series storage)
- Smart meter with IEC 62056 / DLMS/COSEM support (or use the Pecan Street dataset for evaluation)

### Installation

```bash
git clone https://github.com/Joy-Majumder/EnerMind-An-Adaptive-AI-System-for-Personalized-Energy-Budget-Planning.git
cd EnerMind-An-Adaptive-AI-System-for-Personalized-Energy-Budget-Planning
pip install -r requirements.txt
```

### Core Dependencies

```
tensorflow==2.8
scikit-learn==1.0
pandas
numpy
influxdb-client
kafka-python
```

### Running the Prototype

```bash
# Preprocess raw meter data
python src/preprocess.py --data data/pecan_street.csv

# Train the LSTM forecasting model
python src/train_lstm.py --epochs 200 --patience 20

# Start the budget engine and alert service
python src/enermind.py --config config/settings.yaml
```

---

## 📁 Project Structure

```
EnerMind/
├── data/                   # Raw and processed meter data
├── src/
│   ├── preprocess.py       # IQR outlier filtering, spline interpolation, feature engineering
│   ├── train_lstm.py       # LSTM model training and evaluation
│   ├── train_rf.py         # Random Forest cold-start fallback training
│   ├── budget_engine.py    # Personalised quota allocation and adaptive updates
│   ├── alert_service.py    # Threshold evaluation and recommendation generation
│   └── enermind.py         # Main application entry point
├── models/                 # Saved model artefacts
├── notebooks/              # Exploratory analysis and result visualisation
├── config/
│   └── settings.yaml       # System configuration (Kafka, InfluxDB, thresholds)
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

- **Cold-start problem** — full LSTM effectiveness requires ≥ 60 days of historical data; accuracy drops to ~87–89% during the initial period.
- **Hardware dependency** — 15-minute interval smart meters are required for full functionality; monthly-aggregate fallback mode significantly limits alerts and recommendations.
- **Privacy** — smart meter data can reveal occupancy and behavioural patterns; data minimisation and edge processing options are strongly recommended.

---

## 🔭 Future Work

1. **Solar integration** — manage net energy budgets accounting for rooftop PV generation and self-consumption optimisation.
2. **Federated learning** — improve models across households without centralising sensitive data; local gradient aggregation preserves privacy.
3. **Gamification** — household energy challenges, neighbour leaderboards, and reward systems to reinforce long-term behavioural change.
4. **Commercial / industrial extension** — scale EnerMind to departments and production units using the existing modular architecture.

---

## 📚 Citation

If you use EnerMind in your research, please cite:

```bibtex
@article{majumdar2024enermind,
  title     = {EnerMind: An Adaptive AI System for Personalized Energy Budget Planning},
  author    = {Majumdar, Joy Gopal and Hazra, Sneha},
  journal   = {Preprint},
  year      = {2024},
  note      = {Department of Computer Science and Engineering,
               United International University / Daffodil International University,
               Dhaka, Bangladesh}
}
```

---

## 📄 References

1. International Energy Agency (IEA), "Energy Efficiency 2022," IEA Report, Paris, France, 2022.
2. S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.
3. J. Froehlich, L. Findlater, and J. Landay, "The design of eco-feedback technology," in *Proc. ACM CHI*, pp. 1999–2008, 2010.
4. E. Mocanu et al., "Deep learning for estimating building energy consumption," *Sustainable Energy, Grids and Networks*, vol. 6, pp. 91–99, 2016.
5. W. Kong et al., "Short-term residential load forecasting based on LSTM recurrent neural network," *IEEE Trans. Smart Grid*, vol. 10, no. 1, pp. 841–851, 2019.
6. M. H. Albadi and E. F. El-Saadany, "A summary of demand response in electricity markets," *Electr. Power Syst. Res.*, vol. 78, no. 11, pp. 1989–1996, 2008.
7. M. Muratori et al., "A highly resolved modelling technique to simulate residential power demand," *Appl. Energy*, vol. 107, pp. 465–473, 2013.
8. Pecan Street Inc., "Dataport: The World's Largest Energy Data Storehouse." [Online]. Available: https://dataport.pecanstreet.org

---

## 👥 Authors

| Name | Institution | Contact |
|---|---|---|
| Joy Gopal Majumdar | United International University, Dhaka, Bangladesh | jmajumdar2520249@bscse.uiu.ac.bd |
| Sneha Hazra | Daffodil International University, Dhaka, Bangladesh | — |

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.