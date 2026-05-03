"""
Machine Learning models for energy consumption forecasting.
Includes LSTM neural network and Random Forest fallback for cold-start.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import config


class LSTMForecaster:
    """LSTM-based energy consumption forecasting model."""
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = config.LSTM_SEQUENCE_LENGTH
        self.is_trained = False
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: tuple of (sequence_length, num_features)
        """
        self.model = keras.Sequential([
            layers.LSTM(
                config.LSTM_LAYER1_UNITS,
                activation='relu',
                input_shape=input_shape,
                return_sequences=True
            ),
            layers.Dropout(config.LSTM_DROPOUT),
            layers.LSTM(
                config.LSTM_LAYER2_UNITS,
                activation='relu'
            ),
            layers.Dropout(config.LSTM_DROPOUT),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=config.LSTM_LEARNING_RATE)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
    
    def create_sequences(self, data, seq_length):
        """
        Create sequences for LSTM training.
        
        Args:
            data: numpy array of shape (n_samples, n_features)
            seq_length: length of sequences
            
        Returns:
            tuple of (X, y) arrays
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length, 0])  # Predict first column (consumption)
        
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train, validation_split=0.2, epochs=None, verbose=0):
        """
        Train LSTM model.
        
        Args:
            X_train: training features (n_samples, seq_length, n_features)
            y_train: training targets
            validation_split: fraction of data to use for validation
            epochs: number of epochs (uses config default if None)
            verbose: verbosity level
        """
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        
        epochs = epochs or config.LSTM_EPOCHS
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=config.LSTM_EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=config.LSTM_BATCH_SIZE,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        self.is_trained = True
        return history
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: input sequences (n_samples, seq_length, n_features)
            
        Returns:
            predictions (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X, verbose=0).flatten()
    
    def forecast_end_of_month(self, member_data, actual_consumption_so_far):
        """
        Forecast end-of-month consumption for a member.
        
        Args:
            member_data: DataFrame with member's consumption data
            actual_consumption_so_far: cumulative consumption to date (kWh)
            
        Returns:
            dict with forecast results
        """
        if not self.is_trained:
            return None
        
        # Use most recent sequence for prediction
        recent_data = member_data['consumption'].tail(self.sequence_length).values
        
        if len(recent_data) < self.sequence_length:
            return None
        
        # Normalize and prepare for LSTM
        recent_data_scaled = self.scaler.transform(recent_data.reshape(-1, 1)).flatten()
        X = recent_data_scaled.reshape(1, -1, 1)
        
        # Predict remaining consumption
        remaining_prediction = self.predict(X)[0]
        remaining_prediction = self.scaler.inverse_transform(
            np.array([[remaining_prediction]])
        )[0, 0]
        
        # Ensure non-negative
        remaining_prediction = max(remaining_prediction, 0)
        
        return {
            'actual_so_far': actual_consumption_so_far,
            'remaining_predicted': remaining_prediction,
            'projected_total': actual_consumption_so_far + remaining_prediction
        }


class RandomForestForecaster:
    """Random Forest fallback for cold-start scenarios."""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    def build_model(self):
        """Build Random Forest model."""
        self.model = RandomForestRegressor(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            n_jobs=-1,
            random_state=42
        )
    
    def train(self, X_train, y_train):
        """
        Train Random Forest model.
        
        Args:
            X_train: training features
            y_train: training targets
        """
        if self.model is None:
            self.build_model()
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: input features
            
        Returns:
            predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def forecast_end_of_month(self, feature_row, actual_consumption_so_far):
        """
        Forecast end-of-month consumption using Random Forest.
        
        Args:
            feature_row: feature vector for prediction
            actual_consumption_so_far: cumulative consumption (kWh)
            
        Returns:
            dict with forecast results
        """
        if not self.is_trained:
            return None
        
        remaining_prediction = self.predict(feature_row.reshape(1, -1))[0]
        remaining_prediction = max(remaining_prediction, 0)
        
        return {
            'actual_so_far': actual_consumption_so_far,
            'remaining_predicted': remaining_prediction,
            'projected_total': actual_consumption_so_far + remaining_prediction
        }


class HybridForecaster:
    """Hybrid forecaster combining LSTM and Random Forest."""
    
    def __init__(self):
        self.lstm = LSTMForecaster()
        self.rf = RandomForestForecaster()
        self.lstm_confidence = 0.0
        self.rf_confidence = 0.0
    
    def train_lstm(self, X_train, y_train, **kwargs):
        """Train LSTM model."""
        self.lstm.train(X_train, y_train, **kwargs)
        self.lstm_confidence = min(1.0, len(X_train) / 2000)  # Increase with data
    
    def train_rf(self, X_train, y_train):
        """Train Random Forest model."""
        self.rf.train(X_train, y_train)
        self.rf_confidence = min(1.0, len(X_train) / 1000)
    
    def forecast(self, lstm_input, rf_input, actual_so_far, method='blend'):
        """
        Make hybrid forecast.
        
        Args:
            lstm_input: input for LSTM model
            rf_input: input for Random Forest model
            actual_so_far: actual consumption to date
            method: 'blend' (weighted average), 'lstm' (LSTM only), or 'rf' (RF only)
            
        Returns:
            dict with forecast results
        """
        if method == 'blend' and self.lstm.is_trained and self.rf.is_trained:
            lstm_forecast = self.lstm.forecast_end_of_month(lstm_input, actual_so_far)
            rf_forecast = self.rf.forecast_end_of_month(rf_input, actual_so_far)
            
            if lstm_forecast is None or rf_forecast is None:
                # Fall back to whichever is available
                return lstm_forecast or rf_forecast
            
            # Blend predictions with confidence weights
            total_confidence = self.lstm_confidence + self.rf_confidence
            lstm_weight = self.lstm_confidence / total_confidence
            rf_weight = self.rf_confidence / total_confidence
            
            blended_remaining = (
                lstm_forecast['remaining_predicted'] * lstm_weight +
                rf_forecast['remaining_predicted'] * rf_weight
            )
            
            return {
                'actual_so_far': actual_so_far,
                'remaining_predicted': blended_remaining,
                'projected_total': actual_so_far + blended_remaining,
                'lstm_confidence': self.lstm_confidence,
                'rf_confidence': self.rf_confidence
            }
        
        elif method == 'lstm' and self.lstm.is_trained:
            return self.lstm.forecast_end_of_month(lstm_input, actual_so_far)
        
        elif method == 'rf' and self.rf.is_trained:
            return self.rf.forecast_end_of_month(rf_input, actual_so_far)
        
        return None


def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance.
    
    Args:
        y_true: ground truth values
        y_pred: predicted values
        
    Returns:
        dict with evaluation metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate accuracy as percentage within 10% tolerance
    tolerance = 0.1
    accuracy = np.mean(np.abs(y_pred - y_true) <= y_true * tolerance) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'accuracy': accuracy
    }
