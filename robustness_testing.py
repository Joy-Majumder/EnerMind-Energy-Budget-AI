#!/usr/bin/env python3
"""
EnerMind — Robustness Testing (Missing Data + Seasonal Variation)
=================================================================
Addresses Reviewer 1 Comment #5 and Reviewer 2 Comment #5:
  "The evaluation does not sufficiently investigate different household
   conditions, seasonal consumption patterns, irregular usage, occupancy
   changes, or missing data scenarios."

Tests:
  1. Missing-data stress test: inject 5%, 10%, 20% random gaps → re-evaluate
  2. Seasonal-split evaluation: train on months 1-6, test on 7-12 (and vice versa)
  3. Occupancy-change simulation: inject a consumption level shift mid-dataset

Outputs:
  - ROBUSTNESS_TESTING_RESULTS.json

Usage:
    python robustness_testing.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')


# ============================================================================
# DATA
# ============================================================================

def load_daily(csv_path):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['consumption_kwh'] = df['Global_active_power'] * 0.25
    df = df[df['consumption_kwh'] > 0]
    df['date'] = df['timestamp'].dt.date
    daily = df.groupby('date')['consumption_kwh'].sum().reset_index()
    daily.columns = ['date', 'daily_kwh']
    daily['date'] = pd.to_datetime(daily['date'])
    return daily.sort_values('date').reset_index(drop=True)


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


def train_and_eval_mlp(X_train, y_train, X_test, y_test):
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                       solver='adam', learning_rate_init=0.001, batch_size=32,
                       max_iter=300, early_stopping=True, n_iter_no_change=20,
                       random_state=42)
    mlp.fit(X_train, y_train)
    preds = mlp.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    hit_rate = np.mean(np.abs(preds - y_test) / np.maximum(y_test, 1e-9) < 0.10) * 100
    return {'mae': float(mae), 'rmse': float(rmse), 'hit_rate': float(hit_rate)}


# ============================================================================
# TEST 1: MISSING DATA STRESS TEST
# ============================================================================

def missing_data_test(daily_values, gap_rates, rng):
    """
    For each gap_rate: randomly remove that fraction of days,
    forward-fill (mimicking the preprocessing pipeline), and re-evaluate.
    """
    split = int(len(daily_values) * 0.67)
    results = []

    lookback = 30
    for gap_rate in gap_rates:
        # Create degraded copy
        degraded = daily_values.copy()
        n_gaps = int(len(degraded) * gap_rate)
        gap_indices = rng.choice(len(degraded), size=n_gaps, replace=False)
        degraded[gap_indices] = np.nan

        # Forward-fill (simulates the preprocessing pipeline)
        filled = pd.Series(degraded).ffill().bfill().values

        # Re-train and evaluate
        train = filled[:split]
        test = filled[split:]
        X_train, y_train = create_features(train, lookback=lookback)
        combined_test = np.concatenate([train[-lookback:], test])
        X_test, y_test = create_features(combined_test, lookback=lookback)

        if len(X_train) > 0 and len(X_test) > 0:
            metrics = train_and_eval_mlp(X_train, y_train, X_test, y_test)
        else:
            metrics = {'mae': float('nan'), 'rmse': float('nan'), 'hit_rate': 0.0}

        results.append({
            'gap_rate_pct': round(gap_rate * 100, 1),
            'gaps_injected': n_gaps,
            **metrics,
        })

    return results


# ============================================================================
# TEST 2: SEASONAL SPLIT EVALUATION
# ============================================================================

def seasonal_split_test(daily_df):
    """
    Train on months 1-6, test on 7-12 (and vice versa).
    Also: train on Q1+Q2+Q3, test on Q4 (etc.) to check quarter effects.
    """
    daily_df = daily_df.copy()
    daily_df['month'] = daily_df['date'].dt.month
    lookback = 30

    results = []

    # Split A: train months 1-6, test months 7-12
    train_a = daily_df[daily_df['month'] <= 6]['daily_kwh'].values
    test_a = daily_df[daily_df['month'] > 6]['daily_kwh'].values
    if len(train_a) > lookback and len(test_a) > 10:
        X_tr, y_tr = create_features(train_a, lookback=lookback)
        comb_a = np.concatenate([train_a[-lookback:], test_a])
        X_te, y_te = create_features(comb_a, lookback=lookback)
        if len(X_tr) > 0 and len(X_te) > 0:
            m = train_and_eval_mlp(X_tr, y_tr, X_te, y_te)
            results.append({
                'config': 'Train months 1-6, Test months 7-12',
                'train_days': len(train_a),
                'test_days': len(test_a),
                **m,
            })

    # Split B: train months 7-12, test months 1-6
    train_b = daily_df[daily_df['month'] > 6]['daily_kwh'].values
    test_b = daily_df[daily_df['month'] <= 6]['daily_kwh'].values
    if len(train_b) > lookback and len(test_b) > 10:
        X_tr, y_tr = create_features(train_b, lookback=lookback)
        comb_b = np.concatenate([train_b[-lookback:], test_b])
        X_te, y_te = create_features(comb_b, lookback=lookback)
        if len(X_tr) > 0 and len(X_te) > 0:
            m = train_and_eval_mlp(X_tr, y_tr, X_te, y_te)
            results.append({
                'config': 'Train months 7-12, Test months 1-6',
                'train_days': len(train_b),
                'test_days': len(test_b),
                **m,
            })

    # Per-quarter test: train on 3 quarters, test on the 4th
    daily_df['quarter'] = daily_df['date'].dt.quarter
    for test_q in [1, 2, 3, 4]:
        train_q = daily_df[daily_df['quarter'] != test_q]['daily_kwh'].values
        test_q_data = daily_df[daily_df['quarter'] == test_q]['daily_kwh'].values
        if len(train_q) > lookback and len(test_q_data) > 10:
            X_tr, y_tr = create_features(train_q, lookback=lookback)
            comb_q = np.concatenate([train_q[-lookback:], test_q_data])
            X_te, y_te = create_features(comb_q, lookback=lookback)
            if len(X_tr) > 0 and len(X_te) > 0:
                m = train_and_eval_mlp(X_tr, y_tr, X_te, y_te)
                results.append({
                    'config': f'Train Q{{1,2,3,4}}\\Q{test_q}, Test Q{test_q}',
                    'train_days': len(train_q),
                    'test_days': len(test_q_data),
                    **m,
                })

    return results


# ============================================================================
# TEST 3: OCCUPANCY CHANGE (CONSUMPTION LEVEL SHIFT)
# ============================================================================

def occupancy_shift_test(daily_values):
    """
    Simulate a mid-dataset consumption level shift (e.g., new member or
    departure) by scaling the second half by 1.3x and 0.7x.
    """
    split = int(len(daily_values) * 0.67)
    lookback = 30
    results = []

    for scale, label in [(1.3, '+30% occupancy increase'), (0.7, '-30% occupancy decrease')]:
        modified = daily_values.copy()
        midpoint = len(modified) // 2
        modified[midpoint:] = modified[midpoint:] * scale

        train = modified[:split]
        test = modified[split:]
        X_tr, y_tr = create_features(train, lookback=lookback)
        comb = np.concatenate([train[-lookback:], test])
        X_te, y_te = create_features(comb, lookback=lookback)

        if len(X_tr) > 0 and len(X_te) > 0:
            m = train_and_eval_mlp(X_tr, y_tr, X_te, y_te)
            results.append({
                'config': label,
                'scale_factor': scale,
                'shift_at_day': midpoint,
                **m,
            })

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 100)
    print("ENERMIND — ROBUSTNESS TESTING (MISSING DATA + SEASONAL + OCCUPANCY)")
    print("=" * 100 + "\n")

    csv_path = "DataSets/uci_id_235_household_power_consumption_sample.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] Dataset not found: {csv_path}")
        return

    daily_df = load_daily(csv_path)
    values = daily_df['daily_kwh'].values
    rng = np.random.default_rng(42)

    # Baseline
    split = int(len(values) * 0.67)
    X_tr, y_tr = create_features(values[:split], lookback=7)
    X_te, y_te = create_features(values[split:], lookback=7)
    baseline = train_and_eval_mlp(X_tr, y_tr, X_te, y_te)
    print(f"[✓] Baseline (no degradation): MAE={baseline['mae']:.4f} kWh  |  RMSE={baseline['rmse']:.4f} kWh\n")

    # ── Test 1: Missing Data ──
    print("=" * 100)
    print("TEST 1: MISSING-DATA STRESS TEST")
    print("=" * 100 + "\n")

    gap_rates = [0.05, 0.10, 0.20, 0.30]
    missing_results = missing_data_test(values, gap_rates, rng)

    print(f"{'Gap Rate':>10}  {'MAE (kWh)':>12}  {'RMSE (kWh)':>12}  {'Hit Rate':>10}  {'MAE Δ vs Baseline':>18}")
    print("-" * 70)
    print(f"{'0% (base)':>10}  {baseline['mae']:>12.4f}  {baseline['rmse']:>12.4f}  {baseline['hit_rate']:>9.1f}%  {'—':>18}")
    for r in missing_results:
        delta = ((r['mae'] - baseline['mae']) / baseline['mae']) * 100
        print(f"{r['gap_rate_pct']:>9.0f}%  {r['mae']:>12.4f}  {r['rmse']:>12.4f}  {r['hit_rate']:>9.1f}%  {delta:>+17.1f}%")
    print()

    # ── Test 2: Seasonal Split ──
    print("=" * 100)
    print("TEST 2: SEASONAL-SPLIT EVALUATION")
    print("=" * 100 + "\n")

    seasonal_results = seasonal_split_test(daily_df)

    print(f"{'Configuration':<45}  {'Train':>6}  {'Test':>5}  {'MAE':>8}  {'RMSE':>8}  {'HitRate':>8}")
    print("-" * 90)
    for r in seasonal_results:
        print(f"{r['config']:<45}  {r['train_days']:>5}d  {r['test_days']:>4}d  "
              f"{r['mae']:>7.4f}  {r['rmse']:>7.4f}  {r['hit_rate']:>7.1f}%")
    print()

    # ── Test 3: Occupancy Change ──
    print("=" * 100)
    print("TEST 3: OCCUPANCY-CHANGE SIMULATION")
    print("=" * 100 + "\n")

    occupancy_results = occupancy_shift_test(values)

    print(f"{'Scenario':<30}  {'MAE (kWh)':>12}  {'RMSE (kWh)':>12}  {'MAE Δ vs Baseline':>18}")
    print("-" * 80)
    print(f"{'Baseline (no shift)':<30}  {baseline['mae']:>12.4f}  {baseline['rmse']:>12.4f}  {'—':>18}")
    for r in occupancy_results:
        delta = ((r['mae'] - baseline['mae']) / baseline['mae']) * 100
        print(f"{r['config']:<30}  {r['mae']:>12.4f}  {r['rmse']:>12.4f}  {delta:>+17.1f}%")
    print()

    # --- Save ---
    output = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'UCI-235',
        'baseline_metrics': baseline,
        'missing_data_test': missing_results,
        'seasonal_split_test': seasonal_results,
        'occupancy_change_test': occupancy_results,
    }

    out_path = 'ROBUSTNESS_TESTING_RESULTS.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[✓] Results saved to {out_path}\n")


if __name__ == '__main__':
    main()
