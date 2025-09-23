import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor


def test_process_weather_and_features():
    dp = DataPreprocessor()

    # Build a small weather df for one airport with 4 hourly rows
    data = {
        'time': [
            '2025-09-01T00:00:00Z', '2025-09-01T01:00:00Z',
            '2025-09-01T02:00:00Z', '2025-09-01T03:00:00Z'
        ],
        'airport_code': ['AAA'] * 4,
        'temperature_2m': [10.0, 10.5, np.nan, 11.0],
        'relative_humidity_2m': [80, 82, 79, 78],
        'precipitation': [0.0, 0.1, 0.0, 0.0],
        'wind_speed_10m': [5.0, 6.0, 5.5, 4.0],
        'wind_direction_10m': [0, 90, 180, 270],
        'latitude': [59.9] * 4,
        'longitude': [10.7] * 4
    }

    dfw = pd.DataFrame(data)

    cleaned = dp.process_weather(dfw)
    assert 'temperature' in cleaned.columns
    assert cleaned['datetime'].dtype == 'datetime64[ns, UTC]'

    features = dp.feature_engineer_weather(cleaned)
    # Temporal features
    assert 'hour' in features.columns
    assert 'hour_sin' in features.columns and 'hour_cos' in features.columns

    # Rolling and diff
    assert 'temp_diff_1h' in features.columns
    assert 'temp_roll_3h' in features.columns

    # Wind components
    assert 'wind_u' in features.columns and 'wind_v' in features.columns
    # Check simple numeric expectations: headwind/crosswind for north wind on bearing east (90 deg)
    # For wind_direction=90 (from east), wind_heading math: theta = 90 - wd => 0 deg -> u positive
    row = features.iloc[1]
    assert np.isfinite(row['wind_u'])


def test_merge_weather_with_flights_and_bearing():
    dp = DataPreprocessor()

    # Weather with two hours
    w = pd.DataFrame({
        'datetime': pd.to_datetime(['2025-09-01T10:00:00Z', '2025-09-01T11:00:00Z']),
        'airport': ['AAA', 'AAA'],
        'temperature': [12.0, 13.0],
        'wind_speed': [5.0, 5.0],
        'wind_direction': [0, 180],
        'latitude': [59.9, 59.9],
        'longitude': [10.7, 10.7]
    })

    flights = pd.DataFrame({
        'flight_id': ['F1', 'F2'],
        'std': ['2025-09-01T10:15:00Z', '2025-09-01T11:00:00Z'],
        'origin': ['AAA', 'AAA'],
        'destination': ['BBB', 'CCC']
    })

    airports = pd.DataFrame({
        'iata_code': ['AAA', 'BBB', 'CCC'],
        'latitude': [59.9, 60.0, 59.8],
        'longitude': [10.7, 11.0, 10.4]
    })

    merged = dp.merge_weather_with_flights(w, flights, df_airports=airports, flight_time_col='std')
    # Should have joined weather for both flights
    assert 'temperature' in merged.columns
    # Check that flight_bearing_deg exists
    assert 'flight_bearing_deg' in merged.columns
    # Wind head/cross should be present
    # (may be NaN for some rows if bearing cannot be computed)
    assert 'wind_headwind' in merged.columns and 'wind_crosswind' in merged.columns
