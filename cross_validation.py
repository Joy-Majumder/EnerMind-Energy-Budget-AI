#!/usr/bin/env python3
"""
EnerMind — Rolling-Origin Cross-Validation & Paired Statistical Testing
========================================================================
Addresses Reviewer 1 Comment #6 and Reviewer 2 Comment #4:
  "The experimental comparison does not provide sufficiently strong
   evidence. Numerical comparisons should be more statistically supported."

Implements:
  1. Walk-forward (rolling-origin) cross-validation with 5 chronological folds
  2. Mean ± std of MAE and RMSE across folds
  3. Wilcoxon signed-rank paired test between top-2 models
  4. Effect size (rank-biserial correlation)

Outputs:
  - CROSS_VALIDATION_RESULTS.json

Usage:
    python cross_validation.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import wilcoxon
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


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


# ============================================================================
# WALK-FORWARD CROSS-VALIDATION
# ============================================================================

def walk_forward_cv(daily_values, n_folds=5, min_train_days=90):
    """
    Generate walk-forward (expanding-window) train/test splits.
    Each fold adds approximately (total - min_train) / n_folds days to training.
    """
    total = len(daily_values)
    test_block = (total - min_train_days) // n_folds

    folds = []
    for fold_idx in range(n_folds):
        train_end = min_train_days + fold_idx * test_block
        test_end = min(train_end + test_block, total)
        if test_end <= train_end:
            break
        folds.append({
            'fold': fold_idx + 1,
            'train_end': train_end,
            'test_start': train_end,
            'test_end': test_end,
            'train_size': train_end,
            'test_size': test_end - train_end,
        })

    return folds


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Train all available models and return per-day absolute errors."""
    results = {}

    # MLP
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                       solver='adam', learning_rate_init=0.001, batch_size=32,
                       max_iter=300, early_stopping=True, n_iter_no_change=20,
                       random_state=42)
    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_test)
    results['MLP'] = {
        'predictions': mlp_pred,
        'abs_errors': np.abs(y_test - mlp_pred),
    }

    # XGBoost
    if HAS_XGB:
        xgbm = xgb.XGBRegressor(n_estimators=200, max_depth=6,
                                 learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, random_state=42,
                                 verbosity=0)
        xgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        xgb_pred = xgbm.predict(X_test)
        results['XGBoost'] = {
            'predictions': xgb_pred,
            'abs_errors': np.abs(y_test - xgb_pred),
        }

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                               min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['Random Forest'] = {
        'predictions': rf_pred,
        'abs_errors': np.abs(y_test - rf_pred),
    }

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results['Linear Regression'] = {
        'predictions': lr_pred,
        'abs_errors': np.abs(y_test - lr_pred),
    }

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 100)
    print("ENERMIND — ROLLING-ORIGIN CROSS-VALIDATION & PAIRED STATISTICAL TESTING")
    print("=" * 100 + "\n")

    csv_path = "DataSets/uci_id_235_household_power_consumption_sample.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] Dataset not found: {csv_path}")
        return

    daily = load_daily(csv_path)
    print(f"[✓] Loaded {len(daily)} daily records\n")

    n_folds = 5
    folds = walk_forward_cv(daily, n_folds=n_folds, min_train_days=90)
    print(f"[✓] Generated {len(folds)} walk-forward folds:\n")
    for fd in folds:
        print(f"    Fold {fd['fold']}: train days 1–{fd['train_end']}  |  "
              f"test days {fd['test_start']+1}–{fd['test_end']}  "
              f"(train={fd['train_size']}, test={fd['test_size']})")
    print()

    # Collect per-fold metrics
    fold_metrics = {model: {'mae': [], 'rmse': []} for model in
                    ['MLP', 'XGBoost', 'Random Forest', 'Linear Regression']}
    if not HAS_XGB:
        del fold_metrics['XGBoost']

    all_abs_errors = {model: [] for model in fold_metrics}

    lookback = 30
    for fd in folds:
        train_data = daily[:fd['train_end']]
        test_data = daily[fd['test_start']:fd['test_end']]

        X_train, y_train = create_features(train_data, lookback=lookback)
        combined_test = np.concatenate([train_data[-lookback:], test_data])
        X_test, y_test = create_features(combined_test, lookback=lookback)

        if len(X_test) == 0 or len(X_train) == 0:
            print(f"    [SKIP] Fold {fd['fold']}: insufficient data after feature creation.")
            continue

        print(f"  Fold {fd['fold']} — training {len(X_train)} samples, testing {len(X_test)} samples ...")
        results = train_and_evaluate(X_train, y_train, X_test, y_test)

        for model_name, model_res in results.items():
            if model_name not in fold_metrics:
                continue
            mae = mean_absolute_error(y_test[:len(model_res['predictions'])],
                                      model_res['predictions'])
            rmse = np.sqrt(mean_squared_error(
                y_test[:len(model_res['predictions'])], model_res['predictions']))
            fold_metrics[model_name]['mae'].append(mae)
            fold_metrics[model_name]['rmse'].append(rmse)
            all_abs_errors[model_name].extend(model_res['abs_errors'].tolist())

    # --- Summary across folds ---
    print("\n" + "=" * 100)
    print("CROSS-VALIDATION RESULTS — MEAN ± STD ACROSS FOLDS")
    print("=" * 100 + "\n")
    print(f"{'Model':<25} {'MAE (kWh)':>20} {'RMSE (kWh)':>20}")
    print("-" * 67)

    model_mean_mae = {}
    for model_name in ['MLP', 'XGBoost', 'Random Forest', 'Linear Regression']:
        if model_name not in fold_metrics:
            continue
        maes = fold_metrics[model_name]['mae']
        rmses = fold_metrics[model_name]['rmse']
        if not maes:
            continue
        mae_str = f"{np.mean(maes):.4f} ± {np.std(maes):.4f}"
        rmse_str = f"{np.mean(rmses):.4f} ± {np.std(rmses):.4f}"
        print(f"{model_name:<25} {mae_str:>20} {rmse_str:>20}")
        model_mean_mae[model_name] = np.mean(maes)
    print()

    # --- Paired statistical test ---
    print("=" * 100)
    print("WILCOXON SIGNED-RANK PAIRED TEST (TOP-2 MODELS)")
    print("=" * 100 + "\n")

    if len(model_mean_mae) >= 2:
        sorted_models = sorted(model_mean_mae.items(), key=lambda x: x[1])
        model_a, mae_a = sorted_models[0]
        model_b, mae_b = sorted_models[1]

        errors_a = np.array(all_abs_errors[model_a])
        errors_b = np.array(all_abs_errors[model_b])
        min_len = min(len(errors_a), len(errors_b))
        errors_a = errors_a[:min_len]
        errors_b = errors_b[:min_len]

        print(f"  Comparing: {model_a} (mean MAE={mae_a:.4f}) vs {model_b} (mean MAE={mae_b:.4f})")
        print(f"  Paired samples: {min_len}\n")

        if min_len > 10:
            try:
                stat, p_value = wilcoxon(errors_a, errors_b, alternative='two-sided')
                # Effect size: rank-biserial correlation
                n = min_len
                r = 1 - (2 * stat) / (n * (n + 1) / 2)

                print(f"  Wilcoxon W statistic: {stat:.2f}")
                print(f"  p-value:              {p_value:.6f}")
                print(f"  Effect size (r):      {r:.4f}")

                if p_value < 0.05:
                    print(f"\n  ✓ The difference IS statistically significant (p < 0.05).")
                    print(f"    {model_a} is significantly better than {model_b}.")
                else:
                    print(f"\n  ✗ The difference is NOT statistically significant (p = {p_value:.4f} ≥ 0.05).")
                    print(f"    We CANNOT claim {model_a} is superior to {model_b}.")
                    print(f"    This is consistent with the paper's statement that the gap is suggestive")
                    print(f"    rather than conclusive.")

                wilcoxon_result = {
                    'model_a': model_a,
                    'model_b': model_b,
                    'n_samples': int(min_len),
                    'W_statistic': float(stat),
                    'p_value': float(p_value),
                    'effect_size_r': float(r),
                    'significant_at_005': bool(p_value < 0.05),
                }
            except Exception as e:
                print(f"  [WARN] Wilcoxon test failed: {e}")
                wilcoxon_result = {'error': str(e)}
        else:
            print(f"  [SKIP] Too few samples ({min_len}) for Wilcoxon test.")
            wilcoxon_result = {'error': 'insufficient samples'}
    else:
        wilcoxon_result = {'error': 'fewer than 2 models available'}
    print()

    # --- Save ---
    output = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'UCI-235',
        'n_folds': n_folds,
        'folds': folds,
        'cross_validation_results': {},
        'wilcoxon_test': wilcoxon_result,
    }

    for model_name in fold_metrics:
        maes = fold_metrics[model_name]['mae']
        rmses = fold_metrics[model_name]['rmse']
        if maes:
            output['cross_validation_results'][model_name] = {
                'mae_mean': float(np.mean(maes)),
                'mae_std': float(np.std(maes)),
                'rmse_mean': float(np.mean(rmses)),
                'rmse_std': float(np.std(rmses)),
                'per_fold_mae': [float(m) for m in maes],
                'per_fold_rmse': [float(r) for r in rmses],
            }

    out_path = 'CROSS_VALIDATION_RESULTS.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[✓] Results saved to {out_path}\n")


if __name__ == '__main__':
    main()
