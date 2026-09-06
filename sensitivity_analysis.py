#!/usr/bin/env python3
"""
EnerMind — Sensitivity Analysis of Comfort Factor k
====================================================
Addresses Reviewer 1 Comment #3 and Reviewer 2 Comment #3:
  "The budget parameter is not sufficiently justified.
   There is no sensitivity analysis showing how different
   parameter values affect budgets, alert frequency, or savings."

Produces:
  - Theoretical P(exceed) via Gaussian approximation  (Φ-based)
  - Empirical P(exceed) on UCI-235 historical data
  - Alert-frequency estimates for each k
  - SENSITIVITY_ANALYSIS_RESULTS.json

Usage:
    python sensitivity_analysis.py
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm


# ============================================================================
# DATA LOADING
# ============================================================================

def load_daily_data(csv_path):
    """Load and aggregate UCI-235 to daily consumption."""
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


# ============================================================================
# SENSITIVITY SWEEP
# ============================================================================

def run_sensitivity(daily, k_values):
    """
    For each k value, compute:
      - Daily budget B = mean + k * std
      - Monthly budget = B * 30
      - Theoretical P(day exceeds B) = 1 - Φ(k)  [Gaussian approx]
      - Empirical P(day exceeds B) from actual daily data
      - Expected alerts per 30-day month (empirical)
    """
    values = daily['daily_kwh'].values
    mu = values.mean()
    sigma = values.std()
    n_days = len(values)

    results = []
    for k in k_values:
        daily_budget = mu + k * sigma
        monthly_budget = daily_budget * 30

        # Theoretical (Gaussian)
        p_exceed_theoretical = 1.0 - norm.cdf(k)

        # Empirical
        n_exceed = np.sum(values > daily_budget)
        p_exceed_empirical = n_exceed / n_days

        # Expected alerts per month (assuming each exceed-day triggers one alert)
        expected_alerts_per_month = p_exceed_empirical * 30

        results.append({
            'k': k,
            'daily_budget_kwh': round(daily_budget, 3),
            'monthly_budget_kwh': round(monthly_budget, 1),
            'p_exceed_theoretical': round(p_exceed_theoretical * 100, 1),
            'p_exceed_empirical': round(p_exceed_empirical * 100, 1),
            'n_exceed_days': int(n_exceed),
            'total_days': int(n_days),
            'expected_alerts_per_month': round(expected_alerts_per_month, 1),
        })

    return results, mu, sigma


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 100)
    print("ENERMIND — SENSITIVITY ANALYSIS OF COMFORT FACTOR k")
    print("=" * 100 + "\n")

    csv_path = "DataSets/uci_id_235_household_power_consumption_sample.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] Dataset not found: {csv_path}")
        return

    daily = load_daily_data(csv_path)
    print(f"[✓] Loaded {len(daily)} daily records")
    print(f"    Mean daily consumption: {daily['daily_kwh'].mean():.3f} kWh")
    print(f"    Std  daily consumption: {daily['daily_kwh'].std():.3f} kWh\n")

    k_values = [0.00, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
    results, mu, sigma = run_sensitivity(daily, k_values)

    # --- Print table ---
    print("TABLE — EFFECT OF COMFORT FACTOR k ON BUDGET TIGHTNESS AND OVERRUN RATE")
    print("        (UCI-235, Gaussian Approximation vs. Empirical)\n")
    print(f"{'k':>6}  {'Daily B':>10}  {'Monthly B':>10}  {'P(exceed)':>12}  {'P(exceed)':>12}  {'Alerts':>10}")
    print(f"{'':>6}  {'(kWh)':>10}  {'(kWh)':>10}  {'Theoretical':>12}  {'Empirical':>12}  {'per month':>10}")
    print("-" * 75)
    for r in results:
        default = " *" if r['k'] == 0.5 else "  "
        print(f"{r['k']:>5.2f}{default} {r['daily_budget_kwh']:>9.3f}  {r['monthly_budget_kwh']:>10.1f}"
              f"  {r['p_exceed_theoretical']:>10.1f}%  {r['p_exceed_empirical']:>10.1f}%"
              f"  {r['expected_alerts_per_month']:>10.1f}")
    print("-" * 75)
    print("* Default value used in the prototype.\n")

    # --- Interpretation ---
    default_result = [r for r in results if r['k'] == 0.5][0]
    print("[INTERPRETATION]")
    print(f"  At k=0.5 (default): {default_result['p_exceed_empirical']:.1f}% of days empirically")
    print(f"  exceed the daily budget, vs {default_result['p_exceed_theoretical']:.1f}% predicted")
    print(f"  by the Gaussian model. This corresponds to ~{default_result['expected_alerts_per_month']:.0f}")
    print(f"  alert-triggering days per 30-day month.\n")

    tight = [r for r in results if r['k'] == 0.0][0]
    loose = [r for r in results if r['k'] == 2.0][0]
    print(f"  Tightest  (k=0.0): {tight['p_exceed_empirical']:.1f}% overrun → ~{tight['expected_alerts_per_month']:.0f} alerts/month")
    print(f"  Loosest   (k=2.0): {loose['p_exceed_empirical']:.1f}% overrun → ~{loose['expected_alerts_per_month']:.0f} alerts/month")
    print(f"\n  The operator can use this table to select k based on the desired")
    print(f"  balance between alert fatigue (low k) and budget laxity (high k).\n")

    # --- Save ---
    output = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'UCI-235',
        'data_statistics': {
            'mean_daily_kwh': round(mu, 3),
            'std_daily_kwh': round(sigma, 3),
            'n_days': len(daily),
        },
        'k_values_tested': k_values,
        'default_k': 0.5,
        'sensitivity_results': results,
    }

    out_path = 'SENSITIVITY_ANALYSIS_RESULTS.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[✓] Results saved to {out_path}\n")


if __name__ == '__main__':
    main()
