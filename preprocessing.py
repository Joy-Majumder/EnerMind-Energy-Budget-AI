"""
Data preprocessing and feature engineering module.
Handles outlier detection, interpolation, and feature creation.
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import iqr
import config


class DataPreprocessor:
    """Preprocesses raw smart meter data."""
    
    def __init__(self):
        self.outlier_threshold = config.IQR_MULTIPLIER
        self.max_missing_periods = config.MAX_MISSING_PERIODS
    
    def remove_outliers(self, data):
        """
        Remove outliers using Interquartile Range (IQR) method.
        
        Args:
            data: pandas Series of consumption values
            
        Returns:
            pandas Series with outliers replaced via interpolation
        """
        # Calculate rolling median and IQR
        rolling_median = data.rolling(window=12, center=True).median()
        Q1 = data.rolling(window=12).quantile(0.25)
        Q3 = data.rolling(window=12).quantile(0.75)
        IQR_val = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = rolling_median - self.outlier_threshold * IQR_val
        upper_bound = rolling_median + self.outlier_threshold * IQR_val
        
        # Flag outliers
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        
        # Replace outliers with interpolated values
        data_clean = data.copy()
        if outlier_mask.any():
            indices = np.arange(len(data))
            data_clean[outlier_mask] = np.interp(
                indices[outlier_mask],
                indices[~outlier_mask],
                data[~outlier_mask]
            )
        
        return data_clean
    
    def handle_missing_data(self, data):
        """
        Handle missing data with forward-fill, capped at max_missing_periods.
        
        Args:
            data: pandas Series
            
        Returns:
            pandas Series with missing data handled
        """
        data_filled = data.copy()
        
        # Find consecutive missing periods
        missing_groups = data.isna().astype(int).diff().fillna(0)
        group_num = missing_groups.cumsum()
        
        for group in data_filled[data_filled.isna()].groupby(group_num[data_filled.isna()]).groups:
            missing_count = len(data_filled.loc[group_num == group, :].dropna(how='all'))
            
            if missing_count <= self.max_missing_periods:
                # Forward fill for short gaps
                data_filled.fillna(method='ffill', limit=self.max_missing_periods, inplace=True)
            else:
                # Mark long gaps for data quality alert
                pass
        
        return data_filled
    
    def preprocess(self, data):
        """
        Complete preprocessing pipeline.
        
        Args:
            data: pandas Series or DataFrame with consumption data
            
        Returns:
            pandas Series with cleaned data
        """
        # Remove outliers
        data_clean = self.remove_outliers(data)
        
        # Handle missing data
        data_clean = self.handle_missing_data(data_clean)
        
        # Ensure no NaN values remain
        data_clean = data_clean.fillna(data_clean.mean())
        
        return data_clean


class FeatureEngineer:
    """Creates features for ML models."""
    
    def __init__(self):
        self.feature_lags = config.FEATURE_LAGS
        self.rolling_windows = config.ROLLING_WINDOWS
    
    def create_time_features(self, df):
        """
        Create time-based features.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with time features added
        """
        df = df.copy()
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def create_lag_features(self, df, consumption_col='consumption'):
        """
        Create lagged consumption features.
        
        Args:
            df: DataFrame
            consumption_col: name of consumption column
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        period_mult = config.READINGS_PER_DAY if len(df) >= 1000 else 1
        
        for lag in self.feature_lags:
            lag_periods = lag * period_mult  # Convert days to periods
            df[f'consumption_lag_{lag}d'] = df[consumption_col].shift(lag_periods)
        
        return df
    
    def create_rolling_features(self, df, consumption_col='consumption'):
        """
        Create rolling window features.
        
        Args:
            df: DataFrame
            consumption_col: name of consumption column
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        period_mult = config.READINGS_PER_DAY if len(df) >= 1000 else 1
        
        for window in self.rolling_windows:
            window_periods = window * period_mult
            df[f'rolling_mean_{window}d'] = df[consumption_col].rolling(window=window_periods).mean()
            df[f'rolling_std_{window}d'] = df[consumption_col].rolling(window=window_periods).std()
        
        return df
    
    def create_aggregate_features(self, df, consumption_col='consumption'):
        """
        Create hourly and daily aggregate features.
        
        Args:
            df: DataFrame with datetime index
            consumption_col: name of consumption column
            
        Returns:
            DataFrame with aggregate features
        """
        df = df.copy()
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        if len(df) < 1000:
            df['hourly_total'] = df[consumption_col]
            df['daily_total'] = df[consumption_col]
        else:
            # Hourly aggregate
            hourly = df[consumption_col].resample('h').sum()
            df['hourly_total'] = df.index.map(
                hourly.groupby(hourly.index.hour).transform('mean')
            ).values
            
            # Daily aggregate
            df['daily_total'] = df[consumption_col].resample('d').sum().reindex(
                df.index, method='ffill'
            ).values
        
        return df
    
    def engineer_features(self, df, consumption_col='consumption'):
        """
        Complete feature engineering pipeline.
        
        Args:
            df: DataFrame with consumption data
            consumption_col: name of consumption column
            
        Returns:
            DataFrame with all engineered features
        """
        df = self.create_time_features(df)
        df = self.create_lag_features(df, consumption_col)
        df = self.create_rolling_features(df, consumption_col)
        df = self.create_aggregate_features(df, consumption_col)
        
        # Drop rows with NaN values from lag/rolling features
        df = df.dropna()
        
        return df
    
    def get_feature_columns(self):
        """Get list of all engineered feature columns."""
        features = []
        
        # Time features
        features.extend(['hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend'])
        
        # Lag features
        features.extend([f'consumption_lag_{lag}d' for lag in self.feature_lags])
        
        # Rolling features
        for window in self.rolling_windows:
            features.extend([f'rolling_mean_{window}d', f'rolling_std_{window}d'])
        
        # Aggregate features
        features.extend(['hourly_total', 'daily_total'])
        
        return features
