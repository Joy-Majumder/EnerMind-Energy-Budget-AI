"""
Utility functions for EnerMind system.
"""

import json
from datetime import datetime
import pandas as pd


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        return super().default(obj)


def format_alert_message(alert):
    """Format alert for display."""
    return {
        'level': alert['level'],
        'member': alert['member'],
        'message': alert['message'],
        'utilization': f"{alert['utilization_ratio']:.1%}",
        'timestamp': alert['timestamp'].isoformat()
    }


def format_recommendation_message(rec):
    """Format recommendation for display."""
    return {
        'priority': rec['priority'],
        'member': rec['member'],
        'category': rec['category'],
        'suggestion': rec['suggestion'],
        'estimated_savings': f"{rec['estimated_savings']:.2f} kWh",
        'timestamp': rec['timestamp'].isoformat()
    }


def format_forecast(forecast):
    """Format forecast results for display."""
    if not forecast:
        return None
    
    return {
        'actual_so_far': f"{forecast['actual_so_far']:.2f} kWh",
        'remaining_predicted': f"{forecast['remaining_predicted']:.2f} kWh",
        'projected_total': f"{forecast['projected_total']:.2f} kWh",
        'lstm_confidence': f"{forecast.get('lstm_confidence', 0):.2%}",
        'rf_confidence': f"{forecast.get('rf_confidence', 0):.2%}"
    }


def export_report_to_json(report, filename):
    """Export report to JSON file."""
    with open(filename, 'w') as f:
        json.dump(report, f, cls=JSONEncoder, indent=2)


def print_report(report):
    """Pretty print report to console."""
    print("\n" + "=" * 80)
    print("ENERMIND HOUSEHOLD ENERGY REPORT")
    print("=" * 80)
    print(f"Report Generated: {report['timestamp'].isoformat()}\n")
    
    for member_name, member_data in report['members'].items():
        print(f"\n{member_name.upper()}")
        print("-" * 80)
        print(f"  Monthly Budget:        {member_data['budget']:>10.2f} kWh")
        print(f"  Current Consumption:   {member_data['consumption']:>10.2f} kWh")
        print(f"  Projected Total:       {member_data['projected_total']:>10.2f} kWh")
        print(f"  Budget Utilization:    {member_data['utilization']:>10}")
        
        if member_data['alerts']:
            print("\n  ALERTS:")
            for alert in member_data['alerts']:
                level = alert['level']
                utilization = f"{alert['utilization_ratio']:.1%}"
                print(f"    [{level}] {utilization} of budget consumed")
        
        if member_data['recommendations']:
            print("\n  RECOMMENDATIONS:")
            for rec in member_data['recommendations'][:3]:  # Show top 3
                priority = rec['priority']
                suggestion = rec['suggestion']
                savings = f"{rec['estimated_savings']:.1f}"
                print(f"    [{priority}] {suggestion} (~{savings} kWh savings)")
        
        if member_data['recent_daily']:
            print("\n  RECENT DAILY CONSUMPTION (last 7 days):")
            daily_data = member_data['recent_daily']
            for date, consumption in list(daily_data.items())[-7:]:
                print(f"    {date}: {consumption:>6.2f} kWh")
    
    print("\n" + "=" * 80 + "\n")


class PerformanceMetrics:
    """Track system performance metrics."""
    
    def __init__(self):
        self.total_evaluations = 0
        self.alerts_generated = 0
        self.recommendations_generated = 0
        self.total_energy_reduction = 0.0
    
    def record_evaluation(self, evaluation):
        """Record an evaluation."""
        self.total_evaluations += 1
        self.alerts_generated += len(evaluation.get('alerts', []))
        self.recommendations_generated += len(evaluation.get('recommendations', []))
    
    def get_summary(self):
        """Get performance summary."""
        return {
            'total_evaluations': self.total_evaluations,
            'alerts_generated': self.alerts_generated,
            'recommendations_generated': self.recommendations_generated,
            'avg_alerts_per_eval': self.alerts_generated / max(1, self.total_evaluations),
            'avg_recommendations_per_eval': self.recommendations_generated / max(1, self.total_evaluations)
        }
