#!/usr/bin/env python3
"""
EnerMind — Full-Pipeline Replay Evaluation
===========================================
Addresses Reviewer 1 Comment #4 and Reviewer 2 Comment #4:
  "The evaluation focuses mainly on forecasting metrics. The paper
   should evaluate the complete pipeline: forecasting, budget generation,
   alerts, recommendations, and resulting energy reduction."

Replays historical UCI-235 daily traces through the complete pipeline:
  Data → Budget Allocation → Forecasting → Alert Evaluation → Recommendations

Metrics:
  - Alert precision:  % of alert-days where consumption actually exceeded budget
  - Alert recall:     % of actual overrun-days caught by alerts
  - False alarm rate: % of alert-days with no actual overrun
  - Cool-down suppression: % of warranted alerts suppressed by 4-hour CD
  - Recommendation coverage: % of overrun situations with recommendations

Outputs:
  - PIPELINE_EVALUATION_RESULTS.json

Usage:
    python pipeline_evaluation.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

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


# ============================================================================
# PIPELINE COMPONENTS
# ============================================================================

class BudgetPolicy:
    """Member-level budget allocation: B = mean + k * std."""

    def __init__(self, k=0.5):
        self.k = k

    def compute_daily_budget(self, history):
        mu = np.mean(history)
        sigma = np.std(history)
        return mu + self.k * sigma

    def compute_monthly_budget(self, history):
        return self.compute_daily_budget(history) * 30


class AlertEvaluator:
    """Event-driven alert evaluation with cool-down."""

    WARNING_THRESHOLD = 0.80
    CRITICAL_THRESHOLD = 0.95
    COOLDOWN_INTERVALS = 16  # 4 hours ÷ 15 min = 16 intervals; here 1 unit = 1 day

    def __init__(self):
        self.last_alert_day = -999

    def evaluate(self, day_idx, projected_total, monthly_budget):
        """
        Returns alert level and whether it was suppressed by cool-down.
        """
        if monthly_budget <= 0:
            return None, False

        utilization = projected_total / monthly_budget
        alert_level = None
        suppressed = False

        if utilization >= self.CRITICAL_THRESHOLD:
            alert_level = 'CRITICAL'
        elif utilization >= self.WARNING_THRESHOLD:
            alert_level = 'WARNING'

        if alert_level is not None:
            # Check cool-down (in this daily simulation, CD = 1 day minimum gap)
            if (day_idx - self.last_alert_day) < 1:
                suppressed = True
            else:
                self.last_alert_day = day_idx

        return alert_level, suppressed


class RecommendationGenerator:
    """Rule-based recommendation engine."""

    APPLIANCES = {
        'water_heater': {'savings_kwh': 1.5, 'off_peak': True},
        'dishwasher': {'savings_kwh': 0.5, 'off_peak': True},
        'washing_machine': {'savings_kwh': 0.5, 'off_peak': True},
        'dryer': {'savings_kwh': 1.0, 'off_peak': True},
        'ac': {'savings_kwh': 1.5, 'off_peak': False},
    }

    def generate(self, utilization):
        recs = []
        if utilization >= 0.95:
            recs.append({'priority': 'CRITICAL', 'action': 'shift_off_peak',
                         'appliances': [a for a, p in self.APPLIANCES.items() if p['off_peak']]})
            recs.append({'priority': 'CRITICAL', 'action': 'reduce_ac'})
        elif utilization >= 0.80:
            recs.append({'priority': 'MEDIUM', 'action': 'switch_led'})
            recs.append({'priority': 'MEDIUM', 'action': 'schedule_off_peak'})
        return recs


# ============================================================================
# PIPELINE REPLAY
# ============================================================================

def run_pipeline_replay(daily_values, k=0.5, train_ratio=0.67):
    """
    Replay historical data through the full pipeline day by day.
    Train on the first `train_ratio` fraction, then replay test days.
    """
    split = int(len(daily_values) * train_ratio)
    train = daily_values[:split]
    test = daily_values[split:]

    # Train forecaster on training data
    lookback = 30
    X_train, y_train = create_features(train, lookback=lookback)
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                       solver='adam', learning_rate_init=0.001, batch_size=32,
                       max_iter=300, early_stopping=True, n_iter_no_change=20,
                       random_state=42)
    mlp.fit(X_train, y_train)

    # Compute budget from training data
    budget_policy = BudgetPolicy(k=k)
    daily_budget = budget_policy.compute_daily_budget(train)
    monthly_budget = daily_budget * 30

    alert_evaluator = AlertEvaluator()
    rec_generator = RecommendationGenerator()

    # Day-by-day replay
    log = []
    cumulative_kwh = 0.0

    for day_idx in range(len(test)):
        actual_day = test[day_idx]
        cumulative_kwh += actual_day
        day_of_month = (day_idx % 30) + 1  # Simulate 30-day months

        # Reset cumulative at month boundaries
        if day_of_month == 1 and day_idx > 0:
            cumulative_kwh = actual_day

        # Forecast remaining month
        remaining_days = 30 - day_of_month
        history_before = np.concatenate([train, test[:day_idx]])
        if len(history_before) >= lookback + 1:
            X_feat, _ = create_features(history_before, lookback=lookback)
            if len(X_feat) > 0:
                predicted_daily = mlp.predict(X_feat[-1:])[-1]
            else:
                predicted_daily = np.mean(history_before[-7:])
        else:
            predicted_daily = np.mean(train[-30:])

        remaining_predicted = max(0, predicted_daily * remaining_days)
        projected_total = cumulative_kwh + remaining_predicted
        utilization = projected_total / monthly_budget if monthly_budget > 0 else 0

        # Actual overrun?
        actual_overrun = actual_day > daily_budget

        # Alert evaluation
        alert_level, suppressed = alert_evaluator.evaluate(
            day_idx, projected_total, monthly_budget)

        # Recommendations
        recs = rec_generator.generate(utilization) if alert_level else []

        log.append({
            'day_idx': day_idx,
            'day_of_month': day_of_month,
            'actual_kwh': float(actual_day),
            'daily_budget': float(daily_budget),
            'actual_overrun': bool(actual_overrun),
            'cumulative_kwh': float(cumulative_kwh),
            'projected_total': float(projected_total),
            'monthly_budget': float(monthly_budget),
            'utilization': float(utilization),
            'alert_level': alert_level,
            'alert_suppressed': bool(suppressed),
            'n_recommendations': len(recs),
        })

    return log, daily_budget, monthly_budget


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_pipeline_metrics(log):
    """Compute precision, recall, false alarm rate, etc."""
    total_days = len(log)
    overrun_days = [e for e in log if e['actual_overrun']]
    no_overrun_days = [e for e in log if not e['actual_overrun']]

    alert_days = [e for e in log if e['alert_level'] is not None and not e['alert_suppressed']]
    suppressed_days = [e for e in log if e['alert_suppressed']]

    # Alert days that were actual overruns
    true_positives = [e for e in alert_days if e['actual_overrun']]
    # Alert days that were NOT actual overruns
    false_positives = [e for e in alert_days if not e['actual_overrun']]
    # Overrun days NOT caught by alert
    false_negatives = [e for e in overrun_days
                       if e['alert_level'] is None and not e['alert_suppressed']]
    # Suppressed alerts that WERE warranted (actual overrun)
    warranted_suppressed = [e for e in suppressed_days if e['actual_overrun']]

    # Recommendation coverage: overrun days with recommendations
    overrun_with_recs = [e for e in overrun_days if e['n_recommendations'] > 0]

    precision = (len(true_positives) / len(alert_days) * 100) if alert_days else 0
    recall = (len(true_positives) / len(overrun_days) * 100) if overrun_days else 0
    false_alarm_rate = (len(false_positives) / len(alert_days) * 100) if alert_days else 0
    suppression_rate = (len(warranted_suppressed) / max(1, len(suppressed_days))) * 100
    rec_coverage = (len(overrun_with_recs) / len(overrun_days) * 100) if overrun_days else 0

    return {
        'total_days': total_days,
        'overrun_days': len(overrun_days),
        'alert_days_fired': len(alert_days),
        'alerts_suppressed_by_cooldown': len(suppressed_days),
        'true_positives': len(true_positives),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'warranted_alerts_suppressed': len(warranted_suppressed),
        'overrun_days_with_recommendations': len(overrun_with_recs),
        'alert_precision_pct': round(precision, 1),
        'alert_recall_pct': round(recall, 1),
        'false_alarm_rate_pct': round(false_alarm_rate, 1),
        'cooldown_suppression_rate_pct': round(suppression_rate, 1),
        'recommendation_coverage_pct': round(rec_coverage, 1),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 100)
    print("ENERMIND — FULL-PIPELINE REPLAY EVALUATION")
    print("=" * 100 + "\n")

    csv_path = "DataSets/uci_id_235_household_power_consumption_sample.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] Dataset not found: {csv_path}")
        return

    daily = load_daily(csv_path)
    values = daily['daily_kwh'].values
    print(f"[✓] Loaded {len(values)} daily records\n")

    # Run replay with default k=0.5
    log, daily_budget, monthly_budget = run_pipeline_replay(values, k=0.5)
    metrics = compute_pipeline_metrics(log)

    print(f"[✓] Pipeline replay completed: {metrics['total_days']} test days\n")

    print("PIPELINE EVALUATION RESULTS")
    print("-" * 60)
    print(f"  Daily budget (k=0.5):          {daily_budget:.3f} kWh")
    print(f"  Monthly budget:                {monthly_budget:.1f} kWh")
    print(f"  Total test days:               {metrics['total_days']}")
    print(f"  Days with actual overrun:      {metrics['overrun_days']}")
    print()
    print(f"  Alerts fired:                  {metrics['alert_days_fired']}")
    print(f"  Alerts suppressed (cool-down): {metrics['alerts_suppressed_by_cooldown']}")
    print()
    print(f"  Alert precision:               {metrics['alert_precision_pct']:.1f}%")
    print(f"  Alert recall:                  {metrics['alert_recall_pct']:.1f}%")
    print(f"  False alarm rate:              {metrics['false_alarm_rate_pct']:.1f}%")
    print(f"  Cool-down suppression rate:    {metrics['cooldown_suppression_rate_pct']:.1f}%")
    print(f"  Recommendation coverage:       {metrics['recommendation_coverage_pct']:.1f}%")
    print()

    # Also run for multiple k values
    print("=" * 100)
    print("PIPELINE METRICS VS COMFORT FACTOR k")
    print("=" * 100 + "\n")
    print(f"{'k':>6}  {'Overrun':>8}  {'Alerts':>8}  {'Precision':>10}  {'Recall':>8}  {'FalseAlarm':>11}")
    print("-" * 60)

    k_sweep = []
    for k in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        log_k, _, _ = run_pipeline_replay(values, k=k)
        m = compute_pipeline_metrics(log_k)
        print(f"{k:>5.2f}  {m['overrun_days']:>7}d  {m['alert_days_fired']:>7}d  "
              f"{m['alert_precision_pct']:>9.1f}%  {m['alert_recall_pct']:>7.1f}%  "
              f"{m['false_alarm_rate_pct']:>10.1f}%")
        k_sweep.append({'k': k, **m})
    print()

    # --- Save ---
    output = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'UCI-235',
        'default_k': 0.5,
        'pipeline_metrics_default_k': metrics,
        'k_sweep_pipeline_metrics': k_sweep,
        'daily_budget_kwh': float(daily_budget),
        'monthly_budget_kwh': float(monthly_budget),
    }

    out_path = 'PIPELINE_EVALUATION_RESULTS.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[✓] Results saved to {out_path}\n")


if __name__ == '__main__':
    main()
