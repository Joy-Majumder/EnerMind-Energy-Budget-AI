"""
Main entry point for EnerMind prototype.
Demonstrates the complete system workflow:
1. Data generation/ingestion
2. Model training
3. Budget allocation
4. Forecasting
5. Alert and recommendation generation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import logging

from system import EnerMindSystem
from data_generator import SmartMeterDataGenerator
from utils import print_report, export_report_to_json, PerformanceMetrics
import config


def setup_system():
    """Initialize and setup EnerMind system."""
    print("\n" + "=" * 80)
    print("ENERMIND - Adaptive AI System for Personalized Energy Budget Planning")
    print("=" * 80 + "\n")
    
    # Initialize system
    system = EnerMindSystem()
    print("[✓] EnerMind system initialized")
    
    return system


def generate_and_ingest_data(system):
    """Generate and ingest smart meter data."""
    print("\n[*] Generating smart meter data...")
    
    # Generate 12 months of data for 4 household members
    generator = SmartMeterDataGenerator(seed=42)
    start_date = datetime(2023, 1, 1)
    
    # Generate 8 months for training, 4 months for testing
    data = generator.generate_household_data(start_date, num_days=365)
    
    print(f"    Generated {len(data)} meter readings")
    print(f"    Date range: {data['timestamp'].min().date()} to {data['timestamp'].max().date()}")
    print(f"    Household members: {', '.join(set(data['member']))}")
    
    # Aggregate to daily consumption per member
    daily_data = data.groupby(['member', pd.Grouper(key='timestamp', freq='D')]).agg({
        'consumption_kwh': 'sum'
    }).reset_index()
    
    # Ingest data for each member
    for member_name in set(data['member']):
        member_daily = daily_data[daily_data['member'] == member_name].copy()
        member_daily = member_daily.set_index('timestamp')
        member_daily = member_daily[['consumption_kwh']]
        
        system.ingest_member_data(member_name, member_daily)
    
    print("[✓] Data ingested for all household members\n")
    
    return system, data


def initialize_budgets(system):
    """Initialize energy budgets for all members."""
    print("[*] Initializing personalized energy budgets...")
    print("    Formula: Budget = mean(daily_consumption) + k×std(daily_consumption)")
    print(f"    Comfort factor k = {config.COMFORT_FACTOR}\n")
    
    system.initialize_budgets()
    
    # Display budgets
    budgets = system.budget_manager.get_all_budgets()
    for member_name, budget in budgets.items():
        profile = config.HOUSEHOLD_MEMBERS[member_name]
        print(f"    {member_name:12} ({profile['profile']:18}): {budget:>8.2f} kWh/month")
    
    print("\n[✓] Budgets initialized")


def train_forecasting_models(system):
    """Train ML models for consumption forecasting."""
    print("\n[*] Training energy forecasting models...")
    print(f"    Model architecture: Hybrid (LSTM + Random Forest)")
    print(f"    LSTM: {config.LSTM_LAYER1_UNITS} + {config.LSTM_LAYER2_UNITS} units")
    print(f"    Random Forest: {config.RF_N_ESTIMATORS} estimators\n")
    
    try:
        system.train_all_models(verbose=0)
        print("[✓] Models trained successfully")
    except Exception as e:
        print(f"[⚠] Training encountered issues: {e}")
        print("   Continuing with available models...\n")


def simulate_month_consumption(system, data, start_day=240, end_day=330):
    """Simulate a month of consumption and generate evaluations."""
    print(f"\n[*] Simulating consumption from day {start_day} to {end_day}...")
    
    # Get subset of data
    start_date = datetime(2023, 1, 1) + timedelta(days=start_day)
    end_date = datetime(2023, 1, 1) + timedelta(days=end_day)
    
    simulation_data = data[
        (data['timestamp'] >= start_date) & 
        (data['timestamp'] < end_date)
    ].copy()
    
    # Process daily consumption updates
    daily_sim = simulation_data.groupby(['member', pd.Grouper(key='timestamp', freq='D')]).agg({
        'consumption_kwh': 'sum'
    }).reset_index()
    
    # Update consumption tracker
    for _, row in daily_sim.iterrows():
        system.update_consumption(row['member'], row['consumption_kwh'], row['timestamp'])
    
    print(f"    Processed {len(daily_sim)} daily readings\n")
    
    # Run evaluations
    print("[*] Running household evaluation...")
    evaluations = system.evaluate_household()
    
    print(f"    Evaluated {len(evaluations)} household members\n")
    
    return evaluations


def demonstrate_system():
    """Complete demonstration of EnerMind system."""
    
    # Setup
    system = setup_system()
    
    # Data ingestion
    system, data = generate_and_ingest_data(system)
    
    # Budget initialization
    initialize_budgets(system)
    
    # Model training
    train_forecasting_models(system)
    
    # Consumption simulation and evaluation
    print("\n" + "=" * 80)
    print("SIMULATING HOUSEHOLD ENERGY CONSUMPTION AND EVALUATION")
    print("=" * 80)
    
    # Simulate consumption from days 240-330 (approximately 3 months into year)
    evaluations = simulate_month_consumption(system, data, start_day=240, end_day=330)
    
    # Print summary
    print("[*] Household Evaluation Summary:")
    for eval_result in evaluations:
        member = eval_result['member']
        util = eval_result['utilization_ratio']
        budget = eval_result['budget']
        projected = eval_result['projected_total']
        
        # Status indicator
        if util >= config.CRITICAL_THRESHOLD:
            status = "🔴 CRITICAL"
        elif util >= config.WARNING_THRESHOLD:
            status = "🟡 WARNING"
        else:
            status = "🟢 NORMAL"
        
        print(f"    {member:12} {status:12} | Budget: {budget:7.1f} kWh | "
              f"Projected: {projected:7.1f} kWh ({util:.1%})")
    
    # Generate and display report
    print("\n[*] Generating detailed household report...")
    report = system.generate_report()
    
    print_report(report)
    
    # Export report
    report_filename = '/Users/joy0x1/Downloads/Code/Projects/EnerMind/enermind_report.json'
    export_report_to_json(report, report_filename)
    print(f"[✓] Report exported to {report_filename}\n")
    
    # Demonstrate monthly adjustment
    print("=" * 80)
    print("DEMONSTRATING MONTHLY BUDGET ADJUSTMENT")
    print("=" * 80 + "\n")
    
    print("[*] Adjusting budgets based on consumption performance...")
    system.monthly_budget_adjustment()
    
    new_budgets = system.budget_manager.get_all_budgets()
    print("\n    Updated budgets:")
    for member_name, new_budget in new_budgets.items():
        old_budget = system.budget_manager.budget_history[member_name][-2]['budget']
        change = new_budget - old_budget
        change_pct = (change / old_budget * 100) if old_budget > 0 else 0
        direction = "↑" if change > 0 else "↓" if change < 0 else "→"
        print(f"    {member_name:12} {direction} {old_budget:7.1f} → {new_budget:7.1f} kWh "
              f"({change_pct:+.1f}%)")
    
    print("\n" + "=" * 80)
    print("ENERMIND PROTOTYPE DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✓ Smart meter data ingestion and preprocessing")
    print("  ✓ Personalized budget allocation per household member")
    print("  ✓ LSTM + Random Forest hybrid forecasting model")
    print("  ✓ End-of-month consumption prediction")
    print("  ✓ Real-time alert generation for budget overruns")
    print("  ✓ Intelligent recommendations for energy savings")
    print("  ✓ Monthly budget adjustment based on performance")
    print("  ✓ Comprehensive reporting and analytics")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    try:
        demonstrate_system()
    except KeyboardInterrupt:
        print("\n\n[!] Demonstration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
