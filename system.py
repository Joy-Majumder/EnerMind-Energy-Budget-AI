"""
Main EnerMind system - orchestrates all components.
Coordinates data ingestion, ML forecasting, budget management, and alerting.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from preprocessing import DataPreprocessor, FeatureEngineer
from models import HybridForecaster, evaluate_model
from core import BudgetManager, AlertEngine, RecommendationEngine, ConsumptionTracker
from data_generator import SmartMeterDataGenerator
import config


class EnerMindSystem:
    """Main EnerMind system coordinating all modules."""
    
    def __init__(self):
        """Initialize EnerMind system components."""
        self.budget_manager = BudgetManager()
        self.alert_engine = AlertEngine()
        self.recommendation_engine = RecommendationEngine()
        self.consumption_tracker = ConsumptionTracker()
        
        # ML components - one forecaster per household member
        self.forecasters = {}  # member_name -> HybridForecaster
        
        # Data processing
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        
        # Data storage
        self.member_data = {}  # member_name -> DataFrame
        self.training_history = {}  # member_name -> training metrics
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=config.LOG_LEVEL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def ingest_member_data(self, member_name, data_df):
        """
        Ingest consumption data for a household member.
        
        Args:
            member_name: name of household member
            data_df: DataFrame with columns [timestamp, consumption_kwh]
        """
        if not isinstance(data_df.index, pd.DatetimeIndex):
            data_df = data_df.set_index('timestamp')
        
        # Sort by timestamp
        data_df = data_df.sort_index()
        
        # Preprocess data
        data_df['consumption_clean'] = self.preprocessor.preprocess(
            data_df['consumption_kwh']
        )
        
        # Store processed data
        self.member_data[member_name] = data_df
        
        self.logger.info(f"Ingested data for {member_name}: {len(data_df)} records")
    
    def initialize_budgets(self):
        """Initialize budgets for all members based on historical data."""
        for member_name, data_df in self.member_data.items():
            budget = self.budget_manager.calculate_initial_budget(
                member_name,
                data_df['consumption_clean']
            )
            self.logger.info(f"Initialized budget for {member_name}: {budget:.2f} kWh/month")
    
    def prepare_training_data(self, member_name, test_split=0.2):
        """
        Prepare training and test data for a member.
        
        Args:
            member_name: name of household member
            test_split: fraction of data for testing
            
        Returns:
            dict with X_train, X_test, y_train, y_test
        """
        if member_name not in self.member_data:
            raise ValueError(f"No data for member {member_name}")
        
        df = self.member_data[member_name].copy()
        
        # Feature engineering
        df_features = self.feature_engineer.engineer_features(
            df, consumption_col='consumption_clean'
        )
        
        feature_cols = self.feature_engineer.get_feature_columns()
        
        # Prepare sequences for LSTM
        data_values = df_features[['consumption_clean'] + feature_cols].values
        
        # LSTM sequences
        lstm_seq_length = config.LSTM_SEQUENCE_LENGTH * config.READINGS_PER_DAY  # 30 days of data
        X_lstm = []
        y_lstm = []
        
        for i in range(len(data_values) - lstm_seq_length):
            X_lstm.append(data_values[i:i + lstm_seq_length])
            y_lstm.append(data_values[i + lstm_seq_length, 0])
        
        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)
        
        # Train-test split
        split_idx = int(len(X_lstm) * (1 - test_split))
        
        X_train_lstm = X_lstm[:split_idx]
        X_test_lstm = X_lstm[split_idx:]
        y_train_lstm = y_lstm[:split_idx]
        y_test_lstm = y_lstm[split_idx:]
        
        # For Random Forest, use feature vectors
        X_rf = df_features[feature_cols].values
        y_rf = df_features['consumption_clean'].values
        
        X_train_rf = X_rf[:split_idx]
        X_test_rf = X_rf[split_idx:]
        y_train_rf = y_rf[:split_idx]
        y_test_rf = y_rf[split_idx:]
        
        return {
            'X_train_lstm': X_train_lstm,
            'X_test_lstm': X_test_lstm,
            'y_train_lstm': y_train_lstm,
            'y_test_lstm': y_test_lstm,
            'X_train_rf': X_train_rf,
            'X_test_rf': X_test_rf,
            'y_train_rf': y_train_rf,
            'y_test_rf': y_test_rf,
            'feature_cols': feature_cols
        }
    
    def train_member_models(self, member_name, verbose=0):
        """
        Train forecasting models for a member.
        
        Args:
            member_name: name of household member
            verbose: verbosity level for training
        """
        if member_name not in self.member_data:
            raise ValueError(f"No data for member {member_name}")
        
        # Prepare data
        training_data = self.prepare_training_data(member_name)
        
        # Initialize forecaster
        forecaster = HybridForecaster()
        
        # Check if sufficient data for LSTM
        if len(training_data['X_train_lstm']) > 100:
            self.logger.info(f"Training LSTM for {member_name}...")
            history = forecaster.train_lstm(
                training_data['X_train_lstm'],
                training_data['y_train_lstm'],
                epochs=config.LSTM_EPOCHS,
                verbose=verbose
            )
            
            # Evaluate LSTM on test set
            lstm_pred = forecaster.lstm.predict(training_data['X_test_lstm'])
            lstm_metrics = evaluate_model(training_data['y_test_lstm'], lstm_pred)
            self.logger.info(f"LSTM Metrics - MAE: {lstm_metrics['mae']:.4f}, "
                           f"Accuracy: {lstm_metrics['accuracy']:.1f}%")
        else:
            self.logger.warning(f"Insufficient data for LSTM training on {member_name}")
        
        # Train Random Forest (always)
        self.logger.info(f"Training Random Forest for {member_name}...")
        forecaster.train_rf(
            training_data['X_train_rf'],
            training_data['y_train_rf']
        )
        
        # Evaluate Random Forest
        rf_pred = forecaster.rf.predict(training_data['X_test_rf'])
        rf_metrics = evaluate_model(training_data['y_test_rf'], rf_pred)
        self.logger.info(f"RF Metrics - MAE: {rf_metrics['mae']:.4f}, "
                        f"Accuracy: {rf_metrics['accuracy']:.1f}%")
        
        # Store forecaster
        self.forecasters[member_name] = forecaster
        self.training_history[member_name] = {
            'timestamp': datetime.now(),
            'lstm_metrics': lstm_metrics if forecaster.lstm.is_trained else None,
            'rf_metrics': rf_metrics
        }
        
        self.logger.info(f"Models trained for {member_name}")
    
    def train_all_models(self, verbose=0):
        """Train models for all household members."""
        for member_name in self.member_data.keys():
            self.train_member_models(member_name, verbose=verbose)
    
    def update_consumption(self, member_name, consumption_kwh, timestamp=None):
        """
        Record consumption update for a member.
        
        Args:
            member_name: name of household member
            consumption_kwh: consumption reading in kWh
            timestamp: timestamp of reading
        """
        self.consumption_tracker.record_consumption(
            member_name, consumption_kwh, timestamp
        )
    
    def forecast_member_end_of_month(self, member_name):
        """
        Forecast end-of-month consumption for a member.
        
        Args:
            member_name: name of household member
            
        Returns:
            dict with forecast results
        """
        if member_name not in self.forecasters:
            self.logger.warning(f"No trained model for {member_name}")
            return None
        
        if member_name not in self.member_data:
            self.logger.warning(f"No data for {member_name}")
            return None
        
        # Get recent data
        df = self.member_data[member_name]
        actual_consumption = self.consumption_tracker.get_member_consumption(member_name)
        
        # Prepare LSTM input (sequence of recent data)
        recent_data = df['consumption_clean'].tail(
            config.LSTM_SEQUENCE_LENGTH * config.READINGS_PER_DAY
        )
        
        if len(recent_data) < config.LSTM_SEQUENCE_LENGTH * config.READINGS_PER_DAY:
            self.logger.warning(f"Insufficient recent data for {member_name}")
            return None
        
        # Prepare RF input (latest feature vector)
        df_features = self.feature_engineer.engineer_features(
            df.copy(), consumption_col='consumption_clean'
        )
        feature_cols = self.feature_engineer.get_feature_columns()
        latest_features = df_features[feature_cols].iloc[-1].values
        
        # Get forecast
        forecaster = self.forecasters[member_name]
        forecast = forecaster.forecast(
            lstm_input=recent_data,
            rf_input=latest_features,
            actual_so_far=actual_consumption,
            method='blend'
        )
        
        return forecast
    
    def evaluate_member(self, member_name):
        """
        Evaluate budget utilization and generate alerts/recommendations.
        
        Args:
            member_name: name of household member
            
        Returns:
            dict with evaluation results
        """
        budget = self.budget_manager.get_budget(member_name)
        actual_consumption = self.consumption_tracker.get_member_consumption(member_name)
        forecast = self.forecast_member_end_of_month(member_name)
        
        if not forecast:
            return None
        
        projected_total = forecast['projected_total']
        
        # Evaluate alerts
        alerts = self.alert_engine.evaluate_alerts(
            member_name, projected_total, budget
        )
        
        # Generate recommendations
        member_df = self.member_data.get(member_name)
        recommendations = (
            self.recommendation_engine.generate_recommendations(
                member_name, member_df, budget, projected_total
            )
            if member_df is not None else []
        )
        
        utilization_ratio = projected_total / budget if budget > 0 else 0
        
        return {
            'member': member_name,
            'budget': budget,
            'actual_consumption': actual_consumption,
            'projected_total': projected_total,
            'utilization_ratio': utilization_ratio,
            'alerts': alerts,
            'recommendations': recommendations,
            'forecast': forecast,
            'evaluation_time': datetime.now()
        }
    
    def evaluate_household(self):
        """
        Evaluate all household members.
        
        Returns:
            list of evaluation results
        """
        results = []
        
        for member_name in self.member_data.keys():
            evaluation = self.evaluate_member(member_name)
            if evaluation:
                results.append(evaluation)
        
        return results
    
    def generate_report(self, member_name=None):
        """
        Generate comprehensive report for member or household.
        
        Args:
            member_name: specific member or None for all
            
        Returns:
            dict with report data
        """
        if member_name:
            members = [member_name]
        else:
            members = list(self.member_data.keys())
        
        report = {
            'timestamp': datetime.now(),
            'members': {}
        }
        
        for member in members:
            evaluation = self.evaluate_member(member)
            if evaluation:
                report['members'][member] = {
                    'budget': evaluation['budget'],
                    'consumption': evaluation['actual_consumption'],
                    'projected_total': evaluation['projected_total'],
                    'utilization': f"{evaluation['utilization_ratio']:.1%}",
                    'alerts': evaluation['alerts'],
                    'recommendations': evaluation['recommendations'],
                    'recent_daily': self.consumption_tracker.get_daily_consumption(member, days=7)
                }
        
        return report
    
    def monthly_budget_adjustment(self):
        """Adjust budgets at start of new billing cycle."""
        self.logger.info("Starting monthly budget adjustment...")
        
        for member_name in self.member_data.keys():
            actual_consumption = self.consumption_tracker.get_member_consumption(member_name)
            new_budget = self.budget_manager.adjust_budget(
                member_name, actual_consumption, reason='monthly_cycle'
            )
            self.consumption_tracker.reset_member_consumption(member_name)
            
            self.logger.info(
                f"Adjusted budget for {member_name}: {actual_consumption:.2f} kWh used, "
                f"new budget: {new_budget:.2f} kWh"
            )
