"""
H2O.ai Flight Time Prediction Pipeline - HPC Optimized
Predicts actual departure and arrival times using H2O's distributed machine learning platform.
"""

import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from pathlib import Path
import math
import os
import multiprocessing
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DISTANCE AND FLIGHT TIME FEATURES (Same as AutoGluon version)
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance between two points using Haversine formula."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    R = 6371.0  # Earth radius in kilometers
    return R * c


def calculate_flight_time_hours(std: str, sta: str) -> Optional[float]:
    """Calculate expected flight time in hours from scheduled times."""
    try:
        if pd.isna(std) or pd.isna(sta):
            return None
        std_dt = pd.to_datetime(std)
        sta_dt = pd.to_datetime(sta)
        flight_time_hours = (sta_dt - std_dt).total_seconds() / 3600
        if flight_time_hours < 0:
            flight_time_hours += 24
        if flight_time_hours < 0.5 or flight_time_hours > 24:
            return None
        return flight_time_hours
    except Exception:
        return None


def add_distance_and_time_features(df):
    """Add distance and flight time features."""
    airport_coords_df = pd.read_csv("../data/airport_coordinates.csv")
    airport_coords = {}
    valid_coords = airport_coords_df[airport_coords_df['latitude'].notna()]

    for _, row in valid_coords.iterrows():
        airport_coords[row['iata_code']] = {
            'latitude': row['latitude'],
            'longitude': row['longitude']
        }

    df_enhanced = df.copy()
    distances = []
    flight_times = []

    for idx, row in df_enhanced.iterrows():
        if idx % 50000 == 0:
            print(f"   Processed {idx:,} flights...")

        dep_coords = airport_coords.get(row['dep_airport'])
        arr_coords = airport_coords.get(row['arr_airport'])

        if dep_coords and arr_coords:
            distance = haversine_distance(
                dep_coords['latitude'], dep_coords['longitude'],
                arr_coords['latitude'], arr_coords['longitude']
            )
        else:
            distance = None

        distances.append(distance)
        flight_time = calculate_flight_time_hours(row['std'], row['sta'])
        flight_times.append(flight_time)

    df_enhanced['distance_km'] = distances
    df_enhanced['flight_time_hours'] = flight_times
    return df_enhanced


# ============================================================================
# WEATHER FEATURES (Same as AutoGluon version)
# ============================================================================

def process_weather_data(df_weather):
    """Clean and standardize raw weather dataframe."""
    df_clean = df_weather.copy()
    column_mapping = {
        'temperature_2m': 'temperature',
        'relative_humidity_2m': 'humidity',
        'precipitation': 'precipitation',
        'pressure_msl': 'pressure',
        'cloud_cover': 'cloud_cover',
        'wind_speed_10m': 'wind_speed',
        'wind_direction_10m': 'wind_direction',
        'time': 'datetime'
    }

    df_clean = df_clean.rename(columns=column_mapping)
    df_clean['datetime'] = pd.to_datetime(df_clean['datetime'], utc=True).dt.floor('H')

    weather_cols = ['airport', 'datetime', 'temperature', 'humidity', 'precipitation',
                   'pressure', 'cloud_cover', 'wind_speed', 'wind_direction']
    available_cols = [col for col in weather_cols if col in df_clean.columns]
    df_clean = df_clean[available_cols]
    df_clean = df_clean.drop_duplicates(['airport', 'datetime']).sort_values(['airport', 'datetime'])
    return df_clean


def add_weather_features(df_weather):
    """Add derived weather features."""
    df_features = df_weather.copy()
    df_features['hour'] = df_features['datetime'].dt.hour
    df_features['dayofweek'] = df_features['datetime'].dt.dayofweek
    df_features['month'] = df_features['datetime'].dt.month
    df_features['is_weekend'] = df_features['dayofweek'].isin([5, 6]).astype(int)
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)

    if 'wind_speed' in df_features.columns and 'wind_direction' in df_features.columns:
        wind_dir_rad = np.radians(df_features['wind_direction'].fillna(0))
        df_features['wind_u'] = df_features['wind_speed'].fillna(0) * np.cos(wind_dir_rad)
        df_features['wind_v'] = df_features['wind_speed'].fillna(0) * np.sin(wind_dir_rad)
    return df_features


def merge_weather_with_flights(df_flights, df_weather, flight_time_col='std'):
    """Merge weather data with flight data."""
    df_flights_clean = df_flights.copy()
    df_flights_clean['flight_datetime_hour'] = pd.to_datetime(df_flights_clean[flight_time_col], utc=True).dt.floor('H')

    df_merged = df_flights_clean.merge(
        df_weather,
        left_on=['dep_airport', 'flight_datetime_hour'],
        right_on=['airport', 'datetime'],
        how='left',
        suffixes=('', '_weather')
    )

    cols_to_drop = ['airport', 'datetime']
    df_merged = df_merged.drop(columns=[c for c in cols_to_drop if c in df_merged.columns])
    return df_merged


def add_weather_features_to_flights(df_flights):
    """Load weather data and merge with flights."""
    try:
        df_weather_raw = pd.read_csv("../data/weather_data_complete.csv")
    except FileNotFoundError:
        try:
            df_weather_raw = pd.read_csv("../data/weather_data.csv")
        except FileNotFoundError:
            print("No weather data found, skipping weather features")
            return df_flights

    df_weather_clean = process_weather_data(df_weather_raw)
    df_weather_features = add_weather_features(df_weather_clean)
    df_with_weather = merge_weather_with_flights(df_flights, df_weather_features)
    return df_with_weather


# ============================================================================
# TEMPORAL FEATURES (Same as AutoGluon version)
# ============================================================================

def add_temporal_features(df, time_col='std'):
    """Add comprehensive temporal features."""
    df_temporal = df.copy()
    if df_temporal[time_col].dtype == 'object':
        df_temporal[time_col] = pd.to_datetime(df_temporal[time_col], utc=True)

    df_temporal['flight_datetime_hour'] = df_temporal[time_col].dt.floor('h')
    df_temporal['hour'] = df_temporal[time_col].dt.hour
    df_temporal['dayofweek'] = df_temporal[time_col].dt.dayofweek
    df_temporal['month'] = df_temporal[time_col].dt.month
    df_temporal['dayofyear'] = df_temporal[time_col].dt.dayofyear
    df_temporal['week'] = df_temporal[time_col].dt.isocalendar().week
    df_temporal['quarter'] = df_temporal[time_col].dt.quarter
    df_temporal['is_weekend'] = df_temporal['dayofweek'].isin([5, 6]).astype(int)
    df_temporal['is_morning_rush'] = df_temporal['hour'].isin([7, 8, 9]).astype(int)
    df_temporal['is_evening_rush'] = df_temporal['hour'].isin([17, 18, 19]).astype(int)
    df_temporal['is_business_hours'] = df_temporal['hour'].between(9, 17).astype(int)
    return df_temporal


def add_cyclical_features(df):
    """Add cyclical encodings for temporal features."""
    df_cyclical = df.copy()
    if 'hour' in df_cyclical.columns:
        df_cyclical['hour_sin'] = np.sin(2 * np.pi * df_cyclical['hour'] / 24)
        df_cyclical['hour_cos'] = np.cos(2 * np.pi * df_cyclical['hour'] / 24)
    if 'dayofweek' in df_cyclical.columns:
        df_cyclical['dow_sin'] = np.sin(2 * np.pi * df_cyclical['dayofweek'] / 7)
        df_cyclical['dow_cos'] = np.cos(2 * np.pi * df_cyclical['dayofweek'] / 7)
    if 'month' in df_cyclical.columns:
        df_cyclical['month_sin'] = np.sin(2 * np.pi * (df_cyclical['month'] - 1) / 12)
        df_cyclical['month_cos'] = np.cos(2 * np.pi * (df_cyclical['month'] - 1) / 12)
    if 'dayofyear' in df_cyclical.columns:
        df_cyclical['doy_sin'] = np.sin(2 * np.pi * (df_cyclical['dayofyear'] - 1) / 365.25)
        df_cyclical['doy_cos'] = np.cos(2 * np.pi * (df_cyclical['dayofyear'] - 1) / 365.25)
    if 'week' in df_cyclical.columns:
        df_cyclical['week_sin'] = np.sin(2 * np.pi * (df_cyclical['week'] - 1) / 52)
        df_cyclical['week_cos'] = np.cos(2 * np.pi * (df_cyclical['week'] - 1) / 52)
    return df_cyclical


def add_all_temporal_features(df_flights, time_col='std'):
    """Add all temporal features."""
    df_with_temporal = add_temporal_features(df_flights, time_col=time_col)
    df_with_all_temporal = add_cyclical_features(df_with_temporal)
    return df_with_all_temporal


# ============================================================================
# AIRPORT TRAFFIC FEATURES (Same as AutoGluon version)
# ============================================================================

def load_airport_group_mapping():
    """Load airport to group mapping."""
    try:
        df_groups = pd.read_csv("../data/avinor_data/airportgroups.csv")
        airport_to_group = df_groups.set_index('airport')['airport_group'].to_dict()
        return airport_to_group
    except FileNotFoundError:
        print("Airport groups file not found")
        return {}


def load_traffic_data():
    """Load and process scheduled concurrency data."""
    try:
        df_traffic = pd.read_csv("../data/concurrency/scheduled_concurrency.csv")
        df_traffic['datetime'] = pd.to_datetime(df_traffic['date']) + pd.to_timedelta(df_traffic['hour'], unit='h')
        df_traffic['datetime'] = df_traffic['datetime'].dt.floor('h')
        df_traffic = df_traffic[['airport_group', 'datetime', 'number_of_flights']].copy()
        df_traffic = df_traffic.rename(columns={'number_of_flights': 'traffic_count'})
        return df_traffic
    except FileNotFoundError:
        print("Traffic data file not found")
        return pd.DataFrame()


def add_airport_traffic_features(df_flights):
    """Add airport traffic features."""
    airport_to_group = load_airport_group_mapping()
    df_traffic = load_traffic_data()

    if not airport_to_group or df_traffic.empty:
        print("Skipping traffic features - missing data")
        return df_flights

    df_enhanced = df_flights.copy()
    df_enhanced['dep_datetime_hour'] = pd.to_datetime(df_enhanced['std'], utc=True).dt.floor('h')
    df_enhanced['arr_datetime_hour'] = pd.to_datetime(df_enhanced['sta'], utc=True).dt.floor('h')

    traffic_lookup = df_traffic.set_index(['airport_group', 'datetime'])['traffic_count'].to_dict()

    def get_dep_traffic(row):
        airport = row['dep_airport']
        datetime = row['dep_datetime_hour']
        if pd.isna(airport) or pd.isna(datetime) or airport not in airport_to_group:
            return None
        airport_group = airport_to_group[airport]
        return traffic_lookup.get((airport_group, datetime), 0)

    def get_arr_traffic(row):
        airport = row['arr_airport']
        datetime = row['arr_datetime_hour']
        if pd.isna(airport) or pd.isna(datetime) or airport not in airport_to_group:
            return None
        airport_group = airport_to_group[airport]
        return traffic_lookup.get((airport_group, datetime), 0)

    df_enhanced['dep_airport_traffic'] = df_enhanced.apply(get_dep_traffic, axis=1)
    df_enhanced['arr_airport_traffic'] = df_enhanced.apply(get_arr_traffic, axis=1)
    df_enhanced = df_enhanced.drop(columns=['dep_datetime_hour', 'arr_datetime_hour'])

    return df_enhanced


# ============================================================================
# HOLIDAY AND RUSH HOUR FEATURES (Same as AutoGluon version)
# ============================================================================

def add_rush_hour_features(df, time_col='std'):
    """Add rush hour indicators."""
    df_rush = df.copy()
    if df_rush[time_col].dtype == 'object':
        df_rush[time_col] = pd.to_datetime(df_rush[time_col], utc=True)

    hour = df_rush[time_col].dt.hour
    df_rush['is_morning_rush'] = hour.isin([6, 7, 8, 9]).astype(int)
    df_rush['is_evening_rush'] = hour.isin([15, 16, 17, 18, 19, 20]).astype(int)
    df_rush['is_rush_hour'] = (df_rush['is_morning_rush'] | df_rush['is_evening_rush']).astype(int)
    return df_rush


def add_norwegian_holiday_features(df, time_col='std'):
    """Add Norwegian holiday and vacation indicators."""
    df_holiday = df.copy()
    if df_holiday[time_col].dtype == 'object':
        df_holiday[time_col] = pd.to_datetime(df_holiday[time_col], utc=True)

    month = df_holiday[time_col].dt.month
    day = df_holiday[time_col].dt.day
    date_only = df_holiday[time_col].dt.date

    # Load Norwegian public holidays from CSV
    norwegian_public_holidays = set()
    try:
        df_public_holidays = pd.read_csv("../data/helligdager_stavanger.csv")
        df_public_holidays['date'] = pd.to_datetime(df_public_holidays['date']).dt.date
        norwegian_public_holidays = set(df_public_holidays['date'])
        print(f"Loaded {len(norwegian_public_holidays)} Norwegian public holidays")
    except FileNotFoundError:
        print("Norwegian holidays CSV not found, using manual definitions")

    # Manual Norwegian public holidays
    manual_holidays = set()
    for year in range(2018, 2026):
        manual_holidays.update([
            pd.Timestamp(f'{year}-01-01').date(),  # New Year's Day
            pd.Timestamp(f'{year}-05-01').date(),  # Labour Day
            pd.Timestamp(f'{year}-05-17').date(),  # Constitution Day
            pd.Timestamp(f'{year}-12-25').date(),  # Christmas Day
            pd.Timestamp(f'{year}-12-26').date(),  # Boxing Day
        ])

    all_public_holidays = norwegian_public_holidays | manual_holidays
    df_holiday['is_public_holiday'] = date_only.isin(all_public_holidays).astype(int)

    # Norwegian school vacation periods
    df_holiday['is_summer_vacation'] = ((month == 6) & (day >= 15)) | (month == 7) | ((month == 8) & (day <= 15))
    df_holiday['is_winter_vacation'] = ((month == 12) & (day >= 20)) | ((month == 1) & (day <= 8))
    df_holiday['is_winter_sports'] = (month == 2) & (day >= 15) & (day <= 25)
    df_holiday['is_autumn_break'] = (month == 10) & (day >= 1) & (day <= 7)
    df_holiday['is_easter_period'] = ((month == 3) & (day >= 15)) | ((month == 4) & (day <= 15))

    df_holiday['is_school_vacation'] = (
        df_holiday['is_summer_vacation'] |
        df_holiday['is_winter_vacation'] |
        df_holiday['is_winter_sports'] |
        df_holiday['is_autumn_break'] |
        df_holiday['is_easter_period']
    ).astype(int)

    df_holiday['is_holiday_period'] = (
        df_holiday['is_public_holiday'] |
        df_holiday['is_school_vacation']
    ).astype(int)

    # Convert boolean columns to int
    vacation_cols = ['is_summer_vacation', 'is_winter_vacation', 'is_winter_sports',
                    'is_autumn_break', 'is_easter_period']
    for col in vacation_cols:
        df_holiday[col] = df_holiday[col].astype(int)

    return df_holiday


def add_holiday_and_rush_features(df_flights, time_col='std'):
    """Add both holiday and rush hour features."""
    df_with_rush = add_rush_hour_features(df_flights, time_col=time_col)
    df_with_all = add_norwegian_holiday_features(df_with_rush, time_col=time_col)
    return df_with_all


# ============================================================================
# FEATURE ENGINEERING PIPELINE (Same as AutoGluon version)
# ============================================================================

def apply_all_feature_engineering(df_flights):
    """Apply complete feature engineering pipeline."""
    print(f"Starting with: {len(df_flights):,} flights, {len(df_flights.columns)} columns")

    # Step 1: Add distance and flight time features
    print("Adding distance and flight time features...")
    df_step1 = add_distance_and_time_features(df_flights)
    print(f"After distance/time: {len(df_step1.columns)} columns")

    # Step 2: Add weather features
    print("Adding weather features...")
    df_step2 = add_weather_features_to_flights(df_step1)
    print(f"After weather: {len(df_step2.columns)} columns")

    # Step 3: Add temporal features
    print("Adding temporal features...")
    df_step3 = add_all_temporal_features(df_step2)
    print(f"After temporal: {len(df_step3.columns)} columns")

    # Step 4: Add airport traffic features
    print("Adding airport traffic features...")
    df_step4 = add_airport_traffic_features(df_step3)
    print(f"After traffic: {len(df_step4.columns)} columns")

    # Step 5: Add holiday and rush hour features
    print("Adding holiday and rush hour features...")
    df_complete = add_holiday_and_rush_features(df_step4)
    print(f"Final dataset: {df_complete.shape}")
    print(f"Total features added: {len(df_complete.columns) - len(df_flights.columns)}")

    return df_complete


# ============================================================================
# H2O-SPECIFIC DATA PREPARATION
# ============================================================================

def prepare_data_for_h2o(df_complete):
    """Prepare data for H2O modeling."""
    df_model = df_complete.copy()

    # Convert datetime columns and ensure timezone consistency
    for col in ['std', 'sta', 'atd', 'ata']:
        if col in df_model.columns:
            df_model[col] = pd.to_datetime(df_model[col])
            if df_model[col].dt.tz is None:
                df_model[col] = df_model[col].dt.tz_localize('UTC')
            else:
                df_model[col] = df_model[col].dt.tz_convert('UTC')

    # Create FEATURES from scheduled times (these are inputs, not targets)
    df_model['scheduled_departure_timestamp'] = df_model['std'].astype('int64') // 10**9
    df_model['scheduled_arrival_timestamp'] = df_model['sta'].astype('int64') // 10**9
    df_model['scheduled_duration_minutes'] = (df_model['sta'] - df_model['std']).dt.total_seconds() / 60

    # Create TARGET variables (what we want to predict)
    df_model['target_atd'] = df_model['atd'].astype('int64') // 10**9
    df_model['target_ata'] = df_model['ata'].astype('int64') // 10**9

    # Remove rows with missing target variables
    df_model = df_model.dropna(subset=['target_atd', 'target_ata'])

    # Convert categorical columns to strings for H2O
    categorical_cols = ['dep_airport', 'arr_airport']
    for col in categorical_cols:
        if col in df_model.columns:
            df_model[col] = df_model[col].astype(str)

    # Handle infinite values
    df_model = df_model.replace([np.inf, -np.inf], np.nan)

    print(f"Dataset shape for H2O: {df_model.shape}")
    print(f"Target variables: target_atd (actual departure), target_ata (actual arrival)")
    print(f"Key features: scheduled_departure_timestamp, scheduled_arrival_timestamp, scheduled_duration_minutes")
    print(f"Feature columns: {len(df_model.columns) - 2}")

    return df_model


# ============================================================================
# H2O CLUSTER INITIALIZATION AND TRAINING
# ============================================================================

def initialize_h2o_cluster():
    """Initialize H2O cluster with HPC optimization."""
    # Get HPC environment variables
    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

    # Calculate memory allocation (leave some for system)
    available_memory = os.environ.get('SLURM_MEM_PER_NODE', '64G')
    if available_memory.endswith('G'):
        memory_gb = int(available_memory[:-1])
        h2o_memory = f"{max(4, memory_gb - 8)}G"  # Reserve 8GB for system
    else:
        h2o_memory = "32G"  # Default fallback

    print(f"Initializing H2O cluster with {num_cpus} CPUs and {h2o_memory} memory")

    # Initialize H2O cluster
    h2o.init(
        nthreads=num_cpus,
        max_mem_size=h2o_memory,
        strict_version_check=False,
        ignore_config=True
    )

    print(f"H2O cluster initialized successfully")
    print(f"H2O cluster info:")
    print(h2o.cluster().show())


def train_h2o_automl_models(df_model, max_runtime_secs=7200):
    """Train H2O AutoML models for both ATD and ATA prediction."""

    # Initialize H2O cluster
    initialize_h2o_cluster()

    # Convert pandas DataFrame to H2OFrame
    print("Converting data to H2OFrame...")
    h2o_frame = h2o.H2OFrame(df_model)

    # Define feature columns and targets
    feature_columns = [col for col in df_model.columns if col not in ['target_atd', 'target_ata']]

    # Convert categorical columns to factors
    categorical_cols = ['dep_airport', 'arr_airport']
    for col in categorical_cols:
        if col in feature_columns:
            h2o_frame[col] = h2o_frame[col].asfactor()

    print(f"Features: {len(feature_columns)}")
    print(f"Training samples: {h2o_frame.nrows}")

    # Split data into train and test
    train, test = h2o_frame.split_frame(ratios=[0.8], seed=42)
    print(f"Training set: {train.nrows} rows")
    print(f"Test set: {test.nrows} rows")

    # ============================================================================
    # Train ATD (Actual Time of Departure) Model
    # ============================================================================
    print("\n" + "="*60)
    print("Training Actual Time of Departure (ATD) Model")
    print("="*60)

    automl_atd = H2OAutoML(
        max_runtime_secs=max_runtime_secs,
        max_models=20,
        seed=42,
        project_name="avinor_atd_prediction",
        sort_metric="RMSE",
        verbosity="info",
        nfolds=5,
        keep_cross_validation_predictions=True,
        keep_cross_validation_models=True,
        balance_classes=False,
        stopping_metric="RMSE",
        stopping_tolerance=0.001,
        stopping_rounds=3
    )

    automl_atd.train(
        x=feature_columns,
        y='target_atd',
        training_frame=train,
        validation_frame=test
    )

    # Get ATD model performance
    print("\n=== ATD Model Leaderboard ===")
    atd_leaderboard = automl_atd.leaderboard
    print(atd_leaderboard.head())

    atd_best_model = automl_atd.leader
    print(f"\nBest ATD Model: {atd_best_model.model_id}")

    atd_performance = atd_best_model.model_performance(test)
    print(f"ATD Test RMSE: {atd_performance.rmse()}")
    print(f"ATD Test MAE: {atd_performance.mae()}")
    print(f"ATD Test R²: {atd_performance.r2()}")

    # ============================================================================
    # Train ATA (Actual Time of Arrival) Model
    # ============================================================================
    print("\n" + "="*60)
    print("Training Actual Time of Arrival (ATA) Model")
    print("="*60)

    automl_ata = H2OAutoML(
        max_runtime_secs=max_runtime_secs,
        max_models=20,
        seed=42,
        project_name="avinor_ata_prediction",
        sort_metric="RMSE",
        verbosity="info",
        nfolds=5,
        keep_cross_validation_predictions=True,
        keep_cross_validation_models=True,
        balance_classes=False,
        stopping_metric="RMSE",
        stopping_tolerance=0.001,
        stopping_rounds=3
    )

    automl_ata.train(
        x=feature_columns,
        y='target_ata',
        training_frame=train,
        validation_frame=test
    )

    # Get ATA model performance
    print("\n=== ATA Model Leaderboard ===")
    ata_leaderboard = automl_ata.leaderboard
    print(ata_leaderboard.head())

    ata_best_model = automl_ata.leader
    print(f"\nBest ATA Model: {ata_best_model.model_id}")

    ata_performance = ata_best_model.model_performance(test)
    print(f"ATA Test RMSE: {ata_performance.rmse()}")
    print(f"ATA Test MAE: {ata_performance.mae()}")
    print(f"ATA Test R²: {ata_performance.r2()}")

    # ============================================================================
    # Feature Importance and Model Analysis
    # ============================================================================
    print("\n" + "="*60)
    print("Model Analysis and Feature Importance")
    print("="*60)

    # ATD Feature Importance
    print("\n=== ATD Model - Top 15 Feature Importance ===")
    atd_varimp = atd_best_model.varimp(use_pandas=True)
    print(atd_varimp.head(15))

    # ATA Feature Importance
    print("\n=== ATA Model - Top 15 Feature Importance ===")
    ata_varimp = ata_best_model.varimp(use_pandas=True)
    print(ata_varimp.head(15))

    # ============================================================================
    # Save Models
    # ============================================================================
    print("\n=== Saving Models ===")

    # Create model directories
    os.makedirs("./h2o_models_atd/", exist_ok=True)
    os.makedirs("./h2o_models_ata/", exist_ok=True)

    # Save best models
    atd_model_path = h2o.save_model(model=atd_best_model, path="./h2o_models_atd/", force=True)
    ata_model_path = h2o.save_model(model=ata_best_model, path="./h2o_models_ata/", force=True)

    print(f"ATD model saved to: {atd_model_path}")
    print(f"ATA model saved to: {ata_model_path}")

    # Save leaderboards
    atd_leaderboard.as_data_frame().to_csv("./h2o_models_atd/atd_leaderboard.csv", index=False)
    ata_leaderboard.as_data_frame().to_csv("./h2o_models_ata/ata_leaderboard.csv", index=False)

    # Save feature importance
    atd_varimp.to_csv("./h2o_models_atd/atd_feature_importance.csv", index=False)
    ata_varimp.to_csv("./h2o_models_ata/ata_feature_importance.csv", index=False)

    # ============================================================================
    # Sample Predictions
    # ============================================================================
    print("\n=== Sample Predictions ===")

    # Get sample predictions
    sample_data = test.head(5)
    atd_predictions = atd_best_model.predict(sample_data)
    ata_predictions = ata_best_model.predict(sample_data)

    # Convert back to pandas for easier handling
    sample_df = sample_data.as_data_frame()
    atd_pred_df = atd_predictions.as_data_frame()
    ata_pred_df = ata_predictions.as_data_frame()

    for i in range(min(5, len(sample_df))):
        actual_atd = sample_df.iloc[i]['target_atd']
        actual_ata = sample_df.iloc[i]['target_ata']
        pred_atd = atd_pred_df.iloc[i]['predict']
        pred_ata = ata_pred_df.iloc[i]['predict']

        print(f"Flight {i+1}:")
        print(f"  Predicted ATD: {pd.to_datetime(pred_atd, unit='s')}")
        print(f"  Actual ATD:    {pd.to_datetime(actual_atd, unit='s')}")
        print(f"  Predicted ATA: {pd.to_datetime(pred_ata, unit='s')}")
        print(f"  Actual ATA:    {pd.to_datetime(actual_ata, unit='s')}")
        print()

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - Model Performance Summary")
    print("="*60)
    print(f"ATD Model RMSE: {atd_performance.rmse():.4f}")
    print(f"ATD Model MAE:  {atd_performance.mae():.4f}")
    print(f"ATD Model R²:   {atd_performance.r2():.4f}")
    print(f"ATA Model RMSE: {ata_performance.rmse():.4f}")
    print(f"ATA Model MAE:  {ata_performance.mae():.4f}")
    print(f"ATA Model R²:   {ata_performance.r2():.4f}")
    print("="*60)

    return automl_atd, automl_ata, atd_best_model, ata_best_model


def predict_flight_times_h2o(atd_model, ata_model, flight_features_df):
    """Predict both actual departure and arrival times for new flight data using H2O models."""
    # Convert to H2OFrame
    h2o_features = h2o.H2OFrame(flight_features_df)

    # Convert categorical columns to factors
    categorical_cols = ['dep_airport', 'arr_airport']
    for col in categorical_cols:
        if col in flight_features_df.columns:
            h2o_features[col] = h2o_features[col].asfactor()

    # Make predictions
    pred_atd = atd_model.predict(h2o_features)
    pred_ata = ata_model.predict(h2o_features)

    # Convert back to pandas
    pred_atd_df = pred_atd.as_data_frame()
    pred_ata_df = pred_ata.as_data_frame()

    return pred_atd_df['predict'].values, pred_ata_df['predict'].values


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run the complete H2O flight prediction pipeline."""
    print("=" * 60)
    print("H2O.ai Flight Time Prediction Pipeline - HPC Optimized")
    print("=" * 60)

    # Load raw flight data
    data_path = Path("../data")
    df_flights = pd.read_csv(data_path / "avinor_data" / "historical_flights.csv")
    print(f"Loaded {len(df_flights):,} flights with {len(df_flights.columns)} columns")

    # Apply feature engineering
    df_complete = apply_all_feature_engineering(df_flights)

    # Prepare data for H2O modeling
    df_model = prepare_data_for_h2o(df_complete)

    # Train H2O AutoML models
    automl_atd, automl_ata, atd_model, ata_model = train_h2o_automl_models(df_model)

    print("\nH2O models trained and saved successfully!")
    print("Models are ready for deployment!")

    # Shutdown H2O cluster
    h2o.cluster().shutdown()

    return automl_atd, automl_ata, atd_model, ata_model


if __name__ == "__main__":
    models = main()