"""
Smart meter data generator for simulation and testing.
Generates realistic consumption patterns based on household member profiles.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import config


class SmartMeterDataGenerator:
    """Generates realistic smart meter data for household members."""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.members = config.HOUSEHOLD_MEMBERS
    
    def generate_daily_pattern(self, base_consumption, std_dev, date):
        """
        Generate realistic daily consumption pattern with morning and evening peaks.
        
        Args:
            base_consumption: average daily consumption in kWh
            std_dev: standard deviation
            date: date for pattern generation
            
        Returns:
            numpy array of 96 readings (15-minute intervals)
        """
        # Create daily pattern (morning peak 6-9am, evening peak 5-10pm)
        hourly_pattern = np.array([
            0.3, 0.2, 0.15, 0.1, 0.15, 0.4,  # 0-5am
            1.2, 1.5, 1.3, 0.8, 0.6, 0.7,    # 6-11am
            0.8, 0.7, 0.6, 0.65, 0.7, 1.1,   # 12-5pm
            1.8, 2.0, 1.9, 1.6, 1.4, 1.0     # 6-11pm
        ])
        
        # Normalize to base consumption
        hourly_pattern = hourly_pattern / hourly_pattern.sum() * base_consumption
        
        # Generate 15-minute intervals
        readings = []
        for hour in hourly_pattern:
            # Add randomness to each hour
            hour_readings = hour / 4 + np.random.normal(0, std_dev / 20, 4)
            hour_readings = np.maximum(hour_readings, 0)  # No negative consumption
            readings.extend(hour_readings)
        
        return np.array(readings)
    
    def generate_member_data(self, member_name, start_date, num_days, include_anomalies=True):
        """
        Generate consumption data for a household member.
        
        Args:
            member_name: name of household member
            start_date: start date for data generation
            num_days: number of days to generate
            include_anomalies: whether to include realistic anomalies
            
        Returns:
            pandas DataFrame with consumption data
        """
        member_profile = self.members[member_name]
        avg_consumption = member_profile['avg_consumption']
        std_dev = member_profile['std_dev']
        
        data = []
        timestamps = []
        
        for day_offset in range(num_days):
            current_date = start_date + timedelta(days=day_offset)
            
            # Generate base daily consumption
            daily_pattern = self.generate_daily_pattern(avg_consumption, std_dev, current_date)
            
            # Add anomalies (e.g., high usage days, unusual patterns)
            if include_anomalies and np.random.random() < 0.1:  # 10% anomaly rate
                # Random spike
                spike_hour = np.random.randint(0, 24)
                spike_start = spike_hour * 4
                spike_end = min(spike_start + 8, 96)
                daily_pattern[spike_start:spike_end] *= np.random.uniform(1.2, 1.8)
            
            # Generate timestamps for this day
            for interval in range(96):
                timestamp = current_date + timedelta(minutes=interval * 15)
                timestamps.append(timestamp)
                data.append(daily_pattern[interval])
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'member': member_name,
            'consumption_kw': data
        })
        
        # Convert to daily kWh by averaging
        df['consumption_kwh'] = df['consumption_kw'] / 4  # 15-min to hourly to daily
        
        return df
    
    def generate_household_data(self, start_date, num_days, include_anomalies=True):
        """
        Generate consumption data for entire household.
        
        Args:
            start_date: start date for data generation
            num_days: number of days to generate
            include_anomalies: whether to include anomalies
            
        Returns:
            pandas DataFrame with all members' data
        """
        all_data = []
        
        for member_name in self.members.keys():
            member_data = self.generate_member_data(
                member_name, start_date, num_days, include_anomalies
            )
            all_data.append(member_data)
        
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values(['timestamp', 'member']).reset_index(drop=True)
        
        return df
    
    @staticmethod
    def load_pecan_street_simulation(num_days=365):
        """
        Create simulated data mimicking Pecan Street dataset structure.
        
        Args:
            num_days: number of days to simulate
            
        Returns:
            pandas DataFrame
        """
        generator = SmartMeterDataGenerator()
        start_date = datetime(2023, 1, 1)
        
        return generator.generate_household_data(start_date, num_days, include_anomalies=True)
