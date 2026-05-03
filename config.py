"""
Configuration settings for EnerMind system.
"""

# Budget allocation parameters
COMFORT_FACTOR = 0.5  # k factor for budget calculation
BUDGET_LEARNING_RATE = 0.10  # α for budget adjustment
BUDGET_UNDER_REDUCTION = 0.05  # 5% reduction for members under budget
BUDGET_OVER_ADJUSTMENT = 0.03  # 3% increase for members over budget

# Alert thresholds
WARNING_THRESHOLD = 0.80  # 80% of budget
CRITICAL_THRESHOLD = 0.95  # 95% of budget
ALERT_COOLDOWN_MINUTES = 240  # 4 hours

# Data collection parameters
METER_INTERVAL_MINUTES = 15  # 15-minute intervals
READINGS_PER_DAY = 96  # (24 * 60) / 15
MINIMUM_HISTORY_DAYS = 60  # Minimum data for full functionality
COLD_START_DAYS = 60

# LSTM Model parameters
LSTM_LAYER1_UNITS = 128
LSTM_LAYER2_UNITS = 64
LSTM_DROPOUT = 0.2
LSTM_LEARNING_RATE = 0.001
LSTM_BATCH_SIZE = 32
LSTM_EPOCHS = 200
LSTM_EARLY_STOPPING_PATIENCE = 20
LSTM_SEQUENCE_LENGTH = 30  # 30-day historical sequences

# Training parameters
TRAINING_SPLIT_RATIO = 0.8  # 80% train, 20% test
RETRAINING_INTERVAL_WEEKS = 1  # Retrain weekly
RETRAINING_WINDOW_MONTHS = 6  # Use 6 months of recent data

# Random Forest fallback (cold-start)
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 15

# Outlier detection parameters
IQR_MULTIPLIER = 3  # 3 IQR for outlier detection
MAX_MISSING_PERIODS = 6  # 1.5 hours (6 * 15min)

# Household member profiles (for simulation)
HOUSEHOLD_MEMBERS = {
    "Alice": {"avg_consumption": 12.0, "std_dev": 1.5, "profile": "high-consumer"},
    "Bob": {"avg_consumption": 8.0, "std_dev": 1.2, "profile": "medium-consumer"},
    "Charlie": {"avg_consumption": 5.0, "std_dev": 0.8, "profile": "low-consumer"},
    "Diana": {"avg_consumption": 9.0, "std_dev": 3.2, "profile": "variable-consumer"},
}

# Feature engineering parameters
FEATURE_LAGS = [1, 7, 30]  # 1-day, 7-day, 30-day lags
ROLLING_WINDOWS = [7, 30]  # 7-day and 30-day rolling windows

# Database parameters (for future integration)
INFLUX_DB_URL = "http://localhost:8086"
INFLUX_BUCKET = "smart_meter_data"
INFLUX_ORG = "enermind"

# Kafka parameters (for future integration)
KAFKA_BROKERS = ["localhost:9092"]
KAFKA_TOPIC_PREFIX = "enermind"

# Recommendation system
APPLIANCE_PROFILES = {
    "water_heater": {"peak_power": 4.5, "typical_duration": 2.0, "off_peak_compatible": True},
    "dishwasher": {"peak_power": 2.0, "typical_duration": 2.5, "off_peak_compatible": True},
    "washing_machine": {"peak_power": 2.2, "typical_duration": 1.5, "off_peak_compatible": True},
    "dryer": {"peak_power": 5.0, "typical_duration": 1.0, "off_peak_compatible": True},
    "oven": {"peak_power": 3.5, "typical_duration": 1.5, "off_peak_compatible": False},
    "ac": {"peak_power": 3.0, "typical_duration": 4.0, "off_peak_compatible": False},
}

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "logs/enermind.log"
