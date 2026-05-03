"""
Core business logic for EnerMind:
- Budget allocation and adjustment
- Alert and recommendation engine
- Energy consumption tracking
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import config


class BudgetManager:
    """Manages personalized energy budgets for household members."""
    
    def __init__(self):
        self.budgets = {}  # member -> budget (kWh)
        self.budget_history = defaultdict(list)  # member -> list of budgets
        self.performance_history = defaultdict(list)  # member -> list of performance records
    
    def calculate_initial_budget(self, member_name, consumption_data):
        """
        Calculate initial monthly energy budget using Equation 1:
        Bi = mean(Ci) + k * std(Ci)
        
        Args:
            member_name: name of household member
            consumption_data: pandas Series of daily consumption (kWh)
            
        Returns:
            float: monthly budget in kWh
        """
        daily_consumption = consumption_data.resample('D').sum()
        
        mean_consumption = daily_consumption.mean()
        std_consumption = daily_consumption.std()
        
        # Monthly budget = daily budget * 30
        daily_budget = mean_consumption + config.COMFORT_FACTOR * std_consumption
        monthly_budget = daily_budget * 30
        
        self.budgets[member_name] = monthly_budget
        self.budget_history[member_name].append({
            'timestamp': datetime.now(),
            'budget': monthly_budget,
            'reason': 'initial'
        })
        
        return monthly_budget
    
    def adjust_budget(self, member_name, actual_consumption, reason='cycle_adjustment'):
        """
        Adjust budget at start of new billing cycle based on performance.
        
        Args:
            member_name: name of household member
            actual_consumption: actual consumption in previous cycle (kWh)
            reason: reason for adjustment
            
        Returns:
            float: new budget in kWh
        """
        current_budget = self.budgets.get(member_name, 0)
        
        if current_budget == 0:
            return 0
        
        # Calculate adjustment
        if actual_consumption <= current_budget:
            # Member stayed under budget: reduce by a percentage
            reduction = current_budget * config.BUDGET_UNDER_REDUCTION
            new_budget = current_budget - reduction
        else:
            # Member exceeded budget: increase slightly to avoid demotivation
            overage = actual_consumption - current_budget
            adjustment = current_budget * config.BUDGET_OVER_ADJUSTMENT
            new_budget = current_budget + adjustment
        
        self.budgets[member_name] = new_budget
        self.budget_history[member_name].append({
            'timestamp': datetime.now(),
            'budget': new_budget,
            'previous_budget': current_budget,
            'adjustment': new_budget - current_budget,
            'reason': reason
        })
        
        return new_budget
    
    def get_budget(self, member_name):
        """Get current budget for member."""
        return self.budgets.get(member_name, 0)
    
    def get_all_budgets(self):
        """Get all member budgets."""
        return self.budgets.copy()


class AlertEngine:
    """Manages energy consumption alerts."""
    
    def __init__(self):
        self.alerts = defaultdict(list)  # member -> list of alerts
        self.last_alert_time = {}  # member -> last alert timestamp
        self.cooldown_seconds = config.ALERT_COOLDOWN_MINUTES * 60
    
    def evaluate_alerts(self, member_name, projected_total, budget):
        """
        Evaluate alert conditions for member.
        
        Args:
            member_name: name of household member
            projected_total: projected end-of-month consumption (kWh)
            budget: member's energy budget (kWh)
            
        Returns:
            list of alert objects
        """
        alerts = []
        
        if budget == 0:
            return alerts
        
        utilization_ratio = projected_total / budget
        
        # Check cooldown
        last_alert = self.last_alert_time.get(member_name)
        can_alert = (
            last_alert is None or
            (datetime.now() - last_alert).total_seconds() > self.cooldown_seconds
        )
        
        if not can_alert:
            return alerts
        
        # Critical threshold (95% of budget)
        if utilization_ratio >= config.CRITICAL_THRESHOLD:
            alert = {
                'member': member_name,
                'level': 'CRITICAL',
                'timestamp': datetime.now(),
                'utilization_ratio': utilization_ratio,
                'projected_total': projected_total,
                'budget': budget,
                'message': f'CRITICAL: {member_name} projected to use {utilization_ratio:.1%} of monthly budget!'
            }
            alerts.append(alert)
            self.last_alert_time[member_name] = datetime.now()
        
        # Warning threshold (80% of budget)
        elif utilization_ratio >= config.WARNING_THRESHOLD:
            alert = {
                'member': member_name,
                'level': 'WARNING',
                'timestamp': datetime.now(),
                'utilization_ratio': utilization_ratio,
                'projected_total': projected_total,
                'budget': budget,
                'message': f'WARNING: {member_name} projected to use {utilization_ratio:.1%} of monthly budget.'
            }
            alerts.append(alert)
            self.last_alert_time[member_name] = datetime.now()
        
        # Store alerts
        self.alerts[member_name].extend(alerts)
        
        return alerts
    
    def get_alerts(self, member_name=None, limit=None):
        """Get alerts for member or all members."""
        if member_name:
            alerts = self.alerts[member_name]
        else:
            alerts = [a for member_alerts in self.alerts.values() for a in member_alerts]
        
        if limit:
            alerts = alerts[-limit:]
        
        return alerts


class RecommendationEngine:
    """Generates energy-saving recommendations."""
    
    def __init__(self):
        self.recommendations = defaultdict(list)
        self.appliance_profiles = config.APPLIANCE_PROFILES
    
    def generate_recommendations(self, member_name, member_data, budget, projected_total):
        """
        Generate energy-saving recommendations.
        
        Args:
            member_name: name of household member
            member_data: DataFrame with member's consumption data
            budget: member's energy budget
            projected_total: projected end-of-month consumption
            
        Returns:
            list of recommendations
        """
        recommendations = []
        
        if budget == 0:
            return recommendations
        
        utilization_ratio = projected_total / budget
        overage = max(0, projected_total - budget)
        
        if utilization_ratio < 0.80:
            # Member is under budget - encourage current behavior
            recommendation = {
                'member': member_name,
                'priority': 'LOW',
                'category': 'positive_reinforcement',
                'suggestion': f'Great job! You\'re {(1 - utilization_ratio):.0%} under budget. Keep it up!',
                'estimated_savings': 0,
                'timestamp': datetime.now()
            }
            recommendations.append(recommendation)
        
        elif utilization_ratio >= 0.95:
            # Critical - aggressive recommendations
            recommendations.extend(
                self._generate_critical_recommendations(member_name, member_data, overage)
            )
        
        else:
            # Warning level - balanced recommendations
            recommendations.extend(
                self._generate_balanced_recommendations(member_name, member_data, overage)
            )
        
        self.recommendations[member_name].extend(recommendations)
        return recommendations
    
    def _generate_critical_recommendations(self, member_name, member_data, overage):
        """Generate critical-level recommendations."""
        recommendations = []
        
        # Off-peak shifting recommendations
        off_peak_compatible = [
            app for app, profile in self.appliance_profiles.items()
            if profile['off_peak_compatible']
        ]
        
        if off_peak_compatible:
            potential_savings = len(off_peak_compatible) * 0.3  # Estimate 0.3 kWh per appliance
            
            recommendation = {
                'member': member_name,
                'priority': 'CRITICAL',
                'category': 'schedule_shifting',
                'suggestion': f'URGENT: Schedule high-consumption tasks ({", ".join(off_peak_compatible)}) during off-peak hours (9pm-6am) to reduce costs.',
                'estimated_savings': min(potential_savings, overage),
                'timestamp': datetime.now()
            }
            recommendations.append(recommendation)
        
        # Temperature control
        recommendation = {
            'member': member_name,
            'priority': 'CRITICAL',
            'category': 'temperature_control',
            'suggestion': 'Consider reducing AC usage by 1-2 degrees or using a programmable thermostat.',
            'estimated_savings': 1.5,
            'timestamp': datetime.now()
        }
        recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_balanced_recommendations(self, member_name, member_data, overage):
        """Generate balanced-level recommendations."""
        recommendations = []
        
        # LED lighting
        recommendation = {
            'member': member_name,
            'priority': 'MEDIUM',
            'category': 'lighting',
            'suggestion': 'Switch to LED bulbs - they use 75% less energy than incandescent bulbs.',
            'estimated_savings': 0.8,
            'timestamp': datetime.now()
        }
        recommendations.append(recommendation)
        
        # Efficient appliance scheduling
        recommendation = {
            'member': member_name,
            'priority': 'MEDIUM',
            'category': 'appliance_efficiency',
            'suggestion': 'Use dishwasher and washing machine during off-peak hours for lower rates.',
            'estimated_savings': 0.5,
            'timestamp': datetime.now()
        }
        recommendations.append(recommendation)
        
        return recommendations
    
    def get_recommendations(self, member_name=None, limit=5):
        """Get latest recommendations for member or all members."""
        if member_name:
            recs = self.recommendations[member_name]
        else:
            recs = [r for member_recs in self.recommendations.values() for r in member_recs]
        
        # Sort by timestamp (newest first) and apply limit
        recs = sorted(recs, key=lambda x: x['timestamp'], reverse=True)
        
        if limit:
            recs = recs[:limit]
        
        return recs


class ConsumptionTracker:
    """Tracks household consumption across all members."""
    
    def __init__(self):
        self.member_consumption = defaultdict(float)  # member -> total kWh
        self.daily_breakdown = defaultdict(list)  # member -> daily records
        self.consumption_history = defaultdict(list)  # member -> list of records
    
    def record_consumption(self, member_name, consumption_kwh, timestamp=None):
        """Record consumption for a member."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.member_consumption[member_name] += consumption_kwh
        
        record = {
            'timestamp': timestamp,
            'consumption': consumption_kwh
        }
        
        self.consumption_history[member_name].append(record)
    
    def get_member_consumption(self, member_name):
        """Get total consumption for member."""
        return self.member_consumption.get(member_name, 0)
    
    def get_all_consumption(self):
        """Get consumption for all members."""
        return dict(self.member_consumption)
    
    def get_daily_consumption(self, member_name, days=30):
        """Get daily consumption breakdown."""
        all_records = self.consumption_history[member_name]
        
        if not all_records:
            return []
        
        # Group by day
        df = pd.DataFrame(all_records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        daily = df.groupby(df['timestamp'].dt.date)['consumption'].sum()
        
        return daily.tail(days).to_dict()
    
    def reset_member_consumption(self, member_name):
        """Reset monthly consumption counter."""
        self.member_consumption[member_name] = 0
    
    def reset_all_consumption(self):
        """Reset all monthly counters."""
        self.member_consumption.clear()
