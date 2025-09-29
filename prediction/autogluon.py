"""
AutoGluon Flight Time Prediction Pipeline
Predicts actual departure and arrival times using scheduled times and engineered features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import math
from typing import Optional, Tuple
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DISTANCE AND FLIGHT TIME FEATURES
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
# WEATHER FEATURES
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
# TEMPORAL FEATURES
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
# AIRPORT TRAFFIC FEATURES
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
# HOLIDAY AND RUSH HOUR FEATURES
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
# FEATURE ENGINEERING PIPELINE
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
# DATA PREPARATION FOR MODELING
# ============================================================================

def prepare_data_for_modeling(df_complete):
    """Prepare data for AutoGluon modeling."""
    df_model = df_complete.copy()

    # Convert datetime columns and ensure timezone consistency
    # Check if already timezone-aware, if not convert to UTC
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

    print(f"Dataset shape: {df_model.shape}")
    print(f"Target variables: target_atd (actual departure), target_ata (actual arrival)")
    print(f"Key features: scheduled_departure_timestamp, scheduled_arrival_timestamp, scheduled_duration_minutes")
    print(f"Feature columns: {len(df_model.columns) - 2}")

    return df_model


# ============================================================================
# AUTOGLUON TRAINING
# ============================================================================

def train_autogluon_models(df_model, time_limit_per_model=1800):
    """Train AutoGluon models for both ATD and ATA prediction."""

    # Configure for multiprocessing (disable Ray, use native multiprocessing)
    import os
    import multiprocessing
    os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
    os.environ['AUTOGLUON_ENABLE_RAY'] = 'False'

    # Set thread count: use SLURM allocation if available, otherwise use all CPU cores
    num_threads = os.environ.get('SLURM_CPUS_PER_TASK', str(multiprocessing.cpu_count()))
    os.environ['OMP_NUM_THREADS'] = num_threads
    os.environ['MKL_NUM_THREADS'] = num_threads

    # Split data
    feature_columns = [col for col in df_model.columns if col not in ['target_atd', 'target_ata']]
    X = df_model[feature_columns]
    y_atd = df_model['target_atd']
    y_ata = df_model['target_ata']

    # Split for both targets with same random state for consistency
    X_train, X_test, y_atd_train, y_atd_test = train_test_split(X, y_atd, test_size=0.2, random_state=42)
    _, _, y_ata_train, y_ata_test = train_test_split(X, y_ata, test_size=0.2, random_state=42)

    # Create training and test dataframes for AutoGluon
    train_data_atd = X_train.copy()
    train_data_atd['target_atd'] = y_atd_train

    test_data_atd = X_test.copy()
    test_data_atd['target_atd'] = y_atd_test

    train_data_ata = X_train.copy()
    train_data_ata['target_ata'] = y_ata_train

    test_data_ata = X_test.copy()
    test_data_ata['target_ata'] = y_ata_test

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {len(feature_columns)}")

    # Configure AutoGluon predictors with multiprocessing
    predictor_atd = TabularPredictor(
        label='target_atd',
        problem_type='regression',
        eval_metric='root_mean_squared_error',
        path='./autogluon_models_atd/'
    )

    predictor_ata = TabularPredictor(
        label='target_ata',
        problem_type='regression',
        eval_metric='root_mean_squared_error',
        path='./autogluon_models_ata/'
    )

    # Train models
    print("Training Actual Time of Departure (ATD) model...")
    predictor_atd.fit(
        train_data_atd,
        time_limit=time_limit_per_model,
        presets='best_quality',
        verbosity=2
    )

    print("Training Actual Time of Arrival (ATA) model...")
    predictor_ata.fit(
        train_data_ata,
        time_limit=time_limit_per_model,
        presets='best_quality',
        verbosity=2
    )

    # Evaluate models
    print("=== Actual Time of Departure (ATD) Model Performance ===")
    atd_performance = predictor_atd.evaluate(test_data_atd, silent=False)

    print("\n=== Actual Time of Arrival (ATA) Model Performance ===")
    ata_performance = predictor_ata.evaluate(test_data_ata, silent=False)

    print("\n=== Model Performance Summary ===")
    print(f"ATD Model RMSE: {atd_performance}")
    print(f"ATA Model RMSE: {ata_performance}")

    # Feature importance
    print("\n=== ATD Model - Top 10 Feature Importance ===")
    feature_importance_atd = predictor_atd.feature_importance(test_data_atd)
    print(feature_importance_atd.head(10))

    print("\n=== ATA Model - Top 10 Feature Importance ===")
    feature_importance_ata = predictor_ata.feature_importance(test_data_ata)
    print(feature_importance_ata.head(10))

    # Save models
    predictor_atd.save()
    predictor_ata.save()

    return predictor_atd, predictor_ata, test_data_atd, test_data_ata


def predict_flight_times(predictor_atd, predictor_ata, flight_features):
    """Predict both actual departure and arrival times for new flight data."""
    pred_atd = predictor_atd.predict(flight_features)
    pred_ata = predictor_ata.predict(flight_features)
    return pred_atd, pred_ata


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run the complete AutoGluon flight prediction pipeline."""
    print("=" * 60)
    print("AutoGluon Flight Time Prediction Pipeline")
    print("=" * 60)

    # Load raw flight data
    data_path = Path("../data")
    df_flights = pd.read_csv(data_path / "avinor_data" / "historical_flights.csv")
    print(f"Loaded {len(df_flights):,} flights with {len(df_flights.columns)} columns")

    # Apply feature engineering
    df_complete = apply_all_feature_engineering(df_flights)

    # Prepare data for modeling
    df_model = prepare_data_for_modeling(df_complete)

    # Train AutoGluon models
    predictor_atd, predictor_ata, test_data_atd, test_data_ata = train_autogluon_models(df_model)

    # Example predictions
    sample_features = test_data_atd.drop('target_atd', axis=1).head(3)
    pred_atd_sample, pred_ata_sample = predict_flight_times(predictor_atd, predictor_ata, sample_features)

    print("\n=== Sample Predictions ===")
    for i in range(3):
        actual_atd = test_data_atd.iloc[i]['target_atd']
        actual_ata = test_data_ata.iloc[i]['target_ata']
        print(f"Flight {i+1}:")
        print(f"  Predicted ATD: {pd.to_datetime(pred_atd_sample.iloc[i], unit='s')}")
        print(f"  Actual ATD:    {pd.to_datetime(actual_atd, unit='s')}")
        print(f"  Predicted ATA: {pd.to_datetime(pred_ata_sample.iloc[i], unit='s')}")
        print(f"  Actual ATA:    {pd.to_datetime(actual_ata, unit='s')}")
        print()

    print("Models saved and ready for deployment!")
    return predictor_atd, predictor_ata


if __name__ == "__main__":
    predictor_atd, predictor_ata = main()