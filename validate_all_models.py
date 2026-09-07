#!/usr/bin/env python3
"""
EnerMind — Real Model Training & Validation (All 5 Architectures)
=================================================================
Trains MLP, LSTM, XGBoost, Random Forest, and Linear Regression on real
UCI-235 data with a single chronological 67/33 split.

Outputs:
  - REAL_MODEL_RESULTS.json  (Table I numbers + per-day predictions)

This replaces the old train_and_validate.py which fabricated XGBoost and
LSTM results from multipliers.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Optional imports (graceful degradation if not installed)
# ---------------------------------------------------------------------------
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("[WARN] xgboost not available — will skip XGBoost model.")

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except Exception:
    HAS_TF = False
    print("[WARN] tensorflow not available — will skip LSTM model.")



# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_preprocess(csv_path, train_ratio=0.67):
    """Load UCI-235 CSV → daily aggregation → train/test split."""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['consumption_kwh'] = df['Global_active_power'] * 0.25  # 15-min → kWh
    df = df[df['consumption_kwh'] > 0].reset_index(drop=True)

    # Daily aggregation
    df['date'] = df['timestamp'].dt.date
    daily = df.groupby('date').agg({'consumption_kwh': 'sum'}).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date').reset_index(drop=True)

    split = int(len(daily) * train_ratio)
    return daily, split


def create_features(data, lookback=30):
    """Create 6 engineered features strictly from historical data prior to day i."""
    features, targets = [], []
    for i in range(lookback, len(data)):
        f = [
            data[i - 1],                                       # 1-day lag
            data[i - 2] if i >= 2 else data[i - 1],             # 2-day lag
            np.mean(data[max(0, i - 7):i]),                    # 7-day rolling mean
            np.std(data[max(0, i - 7):i]),                     # 7-day rolling std
            np.mean(data[max(0, i - 30):i]),                   # 30-day rolling mean
            np.std(data[max(0, i - 30):i]),                    # 30-day rolling std
        ]
        features.append(f)
        targets.append(data[i])
    return np.array(features), np.array(targets)


def create_lstm_sequences(data, seq_length=30):
    """Create (X, y) sequences for LSTM — predict next day from seq_length days."""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def train_mlp(X_train, y_train, X_test, y_test):
    """MLP (128/64, ReLU) — the deployed prototype model."""
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        batch_size=32,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, model


def train_lstm(train_series, test_series, seq_length=30, epochs=200):
    """LSTM (128/64 stacked layers) — comparison architecture."""
    if not HAS_TF:
        return None, None

    scaler = MinMaxScaler()
    all_data = np.concatenate([train_series, test_series])
    scaler.fit(all_data.reshape(-1, 1))

    train_scaled = scaler.transform(train_series.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_series.reshape(-1, 1)).flatten()

    # Build sequences from train data
    X_train, y_train = create_lstm_sequences(train_scaled, seq_length)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # For test: need seq_length context before each test point
    # Use last seq_length of train + all of test
    combined = np.concatenate([train_scaled[-seq_length:], test_scaled])
    X_test, y_test_scaled = create_lstm_sequences(combined, seq_length)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build model
    model = keras.Sequential([
        layers.LSTM(128, activation='relu', input_shape=(seq_length, 1),
                    return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mae'])

    early_stop = EarlyStopping(monitor='val_loss', patience=20,
                               restore_best_weights=True)

    model.fit(X_train, y_train, batch_size=32, epochs=epochs,
              validation_split=0.2, callbacks=[early_stop], verbose=0)

    preds_scaled = model.predict(X_test, verbose=0).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    return preds, y_true


def train_xgboost(X_train, y_train, X_test, y_test):
    """XGBoost gradient boosting regressor."""
    if not HAS_XGB:
        return None, None
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    preds = model.predict(X_test)
    return preds, model


def train_random_forest(X_train, y_train, X_test, y_test):
    """Random Forest regressor."""
    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_split=5,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, model


def train_linear_regression(X_train, y_train, X_test, y_test):
    """Linear Regression baseline."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, model


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(y_true, y_pred):
    """Compute MAE, RMSE, and hit rate (±10%)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    hit_rate = np.mean(np.abs(y_pred - y_true) / np.maximum(y_true, 1e-9) < 0.10) * 100
    return {'mae': float(mae), 'rmse': float(rmse), 'hit_rate': float(hit_rate)}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 100)
    print("ENERMIND — REAL MODEL TRAINING & VALIDATION (ALL 5 ARCHITECTURES)")
    print("=" * 100 + "\n")

    csv_path = "DataSets/uci_id_235_household_power_consumption_sample.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] Dataset not found: {csv_path}")
        return

    # --- Load data ---
    daily, split = load_and_preprocess(csv_path, train_ratio=0.67)
    values = daily['consumption_kwh'].values
    train_vals = values[:split]
    test_vals = values[split:]

    print(f"[✓] Dataset: {len(daily)} days  |  Train: {len(train_vals)}  |  Test: {len(test_vals)}")
    print(f"    Mean: {values.mean():.3f} kWh/day  |  Std: {values.std():.3f} kWh/day\n")

    # --- Feature-based data (for MLP, XGBoost, RF, LR) ---
    lookback = 30
    X_train_feat, y_train_feat = create_features(train_vals, lookback)
    combined_test = np.concatenate([train_vals[-lookback:], test_vals])
    X_test_feat, y_test_feat = create_features(combined_test, lookback)

    print(f"[✓] Feature matrices — Train: {X_train_feat.shape}  |  Test: {X_test_feat.shape}\n")

    results = {}
    per_day_predictions = {}  # model_name -> list of daily predictions (for paired tests)

    # ---- 1) MLP (Deployed Model) ----
    print("[1/5] Training MLP (128/64, ReLU) ...")
    mlp_preds, _ = train_mlp(X_train_feat, y_train_feat, X_test_feat, y_test_feat)
    results['MLP'] = compute_metrics(y_test_feat, mlp_preds)
    per_day_predictions['MLP'] = mlp_preds.tolist()
    print(f"      MAE: {results['MLP']['mae']:.4f}  |  RMSE: {results['MLP']['rmse']:.4f}  |  Hit Rate: {results['MLP']['hit_rate']:.1f}%\n")

    # ---- 2) LSTM ----
    print("[2/5] Training LSTM (128/64 stacked) ...")
    if HAS_TF:
        lstm_preds, lstm_y_true = train_lstm(train_vals, test_vals, seq_length=30, epochs=200)
        if lstm_preds is not None:
            results['LSTM'] = compute_metrics(lstm_y_true, lstm_preds)
            per_day_predictions['LSTM'] = lstm_preds.tolist()
            print(f"      MAE: {results['LSTM']['mae']:.4f}  |  RMSE: {results['LSTM']['rmse']:.4f}  |  Hit Rate: {results['LSTM']['hit_rate']:.1f}%\n")
        else:
            print("      [SKIP] LSTM training failed.\n")
    else:
        print("      [SKIP] TensorFlow not available.\n")

    # ---- 3) XGBoost ----
    print("[3/5] Training XGBoost ...")
    if HAS_XGB:
        xgb_preds, _ = train_xgboost(X_train_feat, y_train_feat, X_test_feat, y_test_feat)
        if xgb_preds is not None:
            results['XGBoost'] = compute_metrics(y_test_feat, xgb_preds)
            per_day_predictions['XGBoost'] = xgb_preds.tolist()
            print(f"      MAE: {results['XGBoost']['mae']:.4f}  |  RMSE: {results['XGBoost']['rmse']:.4f}  |  Hit Rate: {results['XGBoost']['hit_rate']:.1f}%\n")
        else:
            print("      [SKIP] XGBoost training failed.\n")
    else:
        print("      [SKIP] xgboost not installed.\n")

    # ---- 4) Random Forest ----
    print("[4/5] Training Random Forest ...")
    rf_preds, _ = train_random_forest(X_train_feat, y_train_feat, X_test_feat, y_test_feat)
    results['Random Forest'] = compute_metrics(y_test_feat, rf_preds)
    per_day_predictions['Random Forest'] = rf_preds.tolist()
    print(f"      MAE: {results['Random Forest']['mae']:.4f}  |  RMSE: {results['Random Forest']['rmse']:.4f}  |  Hit Rate: {results['Random Forest']['hit_rate']:.1f}%\n")

    # ---- 5) Linear Regression ----
    print("[5/5] Training Linear Regression ...")
    lr_preds, _ = train_linear_regression(X_train_feat, y_train_feat, X_test_feat, y_test_feat)
    results['Linear Regression'] = compute_metrics(y_test_feat, lr_preds)
    per_day_predictions['Linear Regression'] = lr_preds.tolist()
    print(f"      MAE: {results['Linear Regression']['mae']:.4f}  |  RMSE: {results['Linear Regression']['rmse']:.4f}  |  Hit Rate: {results['Linear Regression']['hit_rate']:.1f}%\n")

    # ---- Summary Table ----
    print("=" * 100)
    print("TABLE I — MODEL FORECASTING PERFORMANCE ON UCI-235")
    print("(Single Household, 365 Days, Single 67/33 Split)")
    print("=" * 100)
    print(f"\n{'Model':<25} {'MAE (kWh)':>12} {'RMSE (kWh)':>12} {'Hit Rate (%)':>14}")
    print("-" * 65)
    for model_name in ['MLP', 'LSTM', 'XGBoost', 'Random Forest', 'Linear Regression']:
        if model_name in results:
            r = results[model_name]
            deployed = " (Deployed)" if model_name == 'MLP' else ""
            print(f"{model_name + deployed:<25} {r['mae']:>12.4f} {r['rmse']:>12.4f} {r['hit_rate']:>14.1f}")
    print("-" * 65)
    print()

    # ---- Save results ----
    output = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'UCI-235 (Individual Household Electric Power Consumption)',
        'total_days': len(daily),
        'train_days': len(train_vals),
        'test_days': len(test_vals),
        'split_ratio': '67/33 chronological',
        'data_statistics': {
            'mean_daily_kwh': float(values.mean()),
            'std_daily_kwh': float(values.std()),
            'min_daily_kwh': float(values.min()),
            'max_daily_kwh': float(values.max()),
        },
        'model_results': results,
        'per_day_predictions': per_day_predictions,
        'ground_truth_test': y_test_feat.tolist(),
    }

    out_path = 'REAL_MODEL_RESULTS.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[✓] Results saved to {out_path}")
    print("[✓] All models trained on REAL data — no estimates or fabricated numbers.\n")


if __name__ == '__main__':
    main()
