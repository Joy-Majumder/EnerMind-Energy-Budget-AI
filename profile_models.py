#!/usr/bin/env python3
"""
EnerMind — Empirical Inference Latency & Model Footprint Profiling
==================================================================
Measures actual single-sample inference latency (over 5,000 iterations),
serialized model footprint (pickle), and raw parameter array sizes for
MLP, Random Forest, and Linear Regression.

Outputs:
  - PROFILING_RESULTS.json

Usage:
    python profile_models.py
"""

import os
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')


def load_daily(csv_path):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['consumption_kwh'] = df['Global_active_power'] * 0.25
    df = df[df['consumption_kwh'] > 0]
    df['date'] = df['timestamp'].dt.date
    daily = df.groupby('date')['consumption_kwh'].sum().values
    return daily


def create_features(data, lookback=30):
    features, targets = [], []
    for i in range(lookback, len(data)):
        f = [
            data[i - 1],
            data[i - 2] if i >= 2 else data[i - 1],
            np.mean(data[max(0, i - 7):i]),
            np.std(data[max(0, i - 7):i]),
            np.mean(data[max(0, i - 30):i]),
            np.std(data[max(0, i - 30):i]),
        ]
        features.append(f)
        targets.append(data[i])
    return np.array(features), np.array(targets)


def main():
    print("\n" + "=" * 100)
    print("ENERMIND — INFERENCE LATENCY & MEMORY FOOTPRINT PROFILING")
    print("=" * 100 + "\n")

    csv_path = "DataSets/uci_id_235_household_power_consumption_sample.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] Dataset not found: {csv_path}")
        return

    daily = load_daily(csv_path)
    split = int(len(daily) * 0.67)
    train_data = daily[:split]
    test_data = daily[split:]
    lookback = 30

    X_train, y_train = create_features(train_data, lookback=lookback)
    comb_test = np.concatenate([train_data[-lookback:], test_data])
    X_test, y_test = create_features(comb_test, lookback=lookback)

    models = {
        'MLP': MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                            solver='adam', learning_rate_init=0.001, batch_size=32,
                            max_iter=300, early_stopping=True, n_iter_no_change=20,
                            random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10,
                                               min_samples_split=5, random_state=42, n_jobs=-1),
        'Linear Regression': LinearRegression()
    }

    single_sample = X_test[:1]
    n_runs = 5000
    profiling_results = {}

    print(f"{'Model':<20} {'Serialized (KB)':>16} {'Raw Params (KB)':>16} {'Mean Latency (ms)':>20} {'p95 Latency (ms)':>20}")
    print("-" * 96)

    for name, model in models.items():
        model.fit(X_train, y_train)

        # Serialized footprint
        serialized = pickle.dumps(model)
        size_bytes = len(serialized)
        size_kb = size_bytes / 1024.0

        # Raw parameter bytes
        param_bytes = 0
        if hasattr(model, 'coefs_'):
            param_bytes = sum(c.nbytes for c in model.coefs_) + sum(i.nbytes for i in model.intercepts_)
        elif hasattr(model, 'coef_'):
            param_bytes = model.coef_.nbytes + (model.intercept_.nbytes if hasattr(model.intercept_, 'nbytes') else 8)

        param_kb = param_bytes / 1024.0

        # Warmup
        for _ in range(100):
            _ = model.predict(single_sample)

        # Timed benchmark
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model.predict(single_sample)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        times = np.array(times)

        profiling_results[name] = {
            'serialized_size_bytes': int(size_bytes),
            'serialized_size_kb': float(round(size_kb, 2)),
            'raw_param_bytes': int(param_bytes),
            'raw_param_kb': float(round(param_kb, 2)),
            'benchmark_iterations': int(n_runs),
            'latency_mean_ms': float(round(np.mean(times), 4)),
            'latency_median_ms': float(round(np.median(times), 4)),
            'latency_p95_ms': float(round(np.percentile(times, 95), 4)),
            'latency_min_ms': float(round(np.min(times), 4)),
            'latency_max_ms': float(round(np.max(times), 4)),
            'latency_std_ms': float(round(np.std(times), 4)),
        }

        print(f"{name:<20} {size_kb:>16.2f} {param_kb:>16.2f} {np.mean(times):>20.4f} {np.percentile(times, 95):>20.4f}")

    print("-" * 96)
    print(f"\n[✓] Benchmarked across {n_runs:,} single-sample inferences per architecture.\n")

    output = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'UCI-235',
        'benchmark_runs_per_model': n_runs,
        'profiling_results': profiling_results
    }

    out_path = 'PROFILING_RESULTS.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[✓] Results saved to {out_path}\n")


if __name__ == '__main__':
    main()
