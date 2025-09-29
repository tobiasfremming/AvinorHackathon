from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import datetime
import requests
from functools import lru_cache
from typing import List, Dict, Tuple
import logging

from src.preprocessing import DataPreprocessor
from src.traffic import compute_concurrency_with_coverage


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Set up logging for holiday functions
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_supported_countries():
    """Get all countries supported by OpenHolidaysAPI."""
    return {
        'albania': 'AL', 'andorra': 'AD', 'austria': 'AT', 'belarus': 'BY', 'belgium': 'BE',
        'brazil': 'BR', 'bulgaria': 'BG', 'croatia': 'HR', 'czechia': 'CZ', 'estonia': 'EE',
        'france': 'FR', 'germany': 'DE', 'hungary': 'HU', 'ireland': 'IE', 'italy': 'IT',
        'latvia': 'LV', 'liechtenstein': 'LI', 'lithuania': 'LT', 'luxembourg': 'LU',
        'malta': 'MT', 'mexico': 'MX', 'moldova': 'MD', 'monaco': 'MC', 'netherlands': 'NL',
        'poland': 'PL', 'portugal': 'PT', 'romania': 'RO', 'san marino': 'SM', 'serbia': 'RS',
        'slovakia': 'SK', 'slovenia': 'SI', 'south africa': 'ZA', 'spain': 'ES',
        'sweden': 'SE', 'switzerland': 'CH', 'vatican city': 'VA',
        # Common aliases
        'czech republic': 'CZ', 'vatican': 'VA', 'norway': 'NO'  # Add Norway support
    }


def fetch_school_holidays(country_code: str, year: int) -> List[Dict]:
    """Fetch school holidays from OpenHolidaysAPI."""
    countries = get_supported_countries()

    # Convert country name to ISO code
    if country_code.lower() in countries:
        iso_code = countries[country_code.lower()]
    else:
        iso_code = country_code.upper()

    url = "https://openholidaysapi.org/SchoolHolidays"
    params = {
        'countryIsoCode': iso_code,
        'languageIsoCode': 'EN',
        'validFrom': f'{year}-01-01',
        'validTo': f'{year}-12-31'
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except:
        return []


@lru_cache(maxsize=128)
def fetch_school_holidays_cached(country_code: str, year: int) -> tuple:
    """
    Cached version of fetch_school_holidays to avoid repeated API calls.
    Returns tuple to make it hashable for caching.
    """
    holidays = fetch_school_holidays(country_code, year)
    # Convert to tuple of tuples for caching
    return tuple((h.get('startDate', ''), h.get('endDate', ''), h.get('name', [{}])[0].get('text', ''))
                 for h in holidays if isinstance(h, dict))


def is_vacation_day(date, country_code: str, buffer_days: int = 2) -> int:
    """
    Given a date and country code, return if the day is in vacation period.
    Uses OpenHolidaysAPI to get actual school holidays for the country.
    Returns 1 if in vacation (includes ±buffer_days buffer), 0 otherwise.

    Args:
        date: Date string or datetime object
        country_code: Country name or ISO code
        buffer_days: Number of buffer days around holidays (default: 2)

    Returns:
        1 if in vacation period (including buffer), 0 otherwise
    """
    try:
        if isinstance(date, str):
            date_obj = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        else:
            date_obj = pd.to_datetime(date).date()

        year = date_obj.year

        # Use cached version
        holidays_tuple = fetch_school_holidays_cached(country_code, year)

        for start_date_str, end_date_str, name in holidays_tuple:
            if not start_date_str or not end_date_str:
                continue

            try:
                start_date = datetime.datetime.strptime(start_date_str[:10], '%Y-%m-%d').date()
                end_date = datetime.datetime.strptime(end_date_str[:10], '%Y-%m-%d').date()

                # Add configurable buffer
                start_buffered = start_date - datetime.timedelta(days=buffer_days)
                end_buffered = end_date + datetime.timedelta(days=buffer_days)

                if start_buffered <= date_obj <= end_buffered:
                    logger.debug(f"Date {date_obj} falls within {name} ({start_date} to {end_date}) with {buffer_days}-day buffer")
                    return 1

            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing dates for holiday {name}: {e}")
                continue

        return 0

    except Exception as e:
        logger.error(f"Error in is_vacation_day for {date}, {country_code}: {e}")
        return 0


def add_holiday_features(
    df: pd.DataFrame,
    date_col: str = 'flight_datetime_hour',
    country_code: str = 'norway',
    buffer_days: int = 2,
) -> pd.DataFrame:
    """
    Add holiday/vacation season features to flight data.

    Args:
        df: DataFrame with flight data
        date_col: Column name containing dates
        country_code: Country to get holiday data for (default: 'norway')
        buffer_days: Buffer days around holidays (default: 2)

    Returns:
        DataFrame with added holiday features
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    if date_col not in out.columns:
        print(f"⚠️ Date column '{date_col}' not found. Skipping holiday features.")
        return out

    print(f"🎄 Adding holiday features for {country_code}...")

    # Ensure date column is datetime
    out[date_col] = pd.to_datetime(out[date_col], utc=True)

    # Extract date for holiday lookup
    out['date_only'] = out[date_col].dt.date

    # Add vacation season indicator
    out['is_vacation_season'] = out['date_only'].apply(
        lambda x: is_vacation_day(x, country_code, buffer_days)
    )

    # If no vacation data found (e.g., for Norway/Sweden), use European default patterns
    if out['is_vacation_season'].sum() == 0:
        print(f"⚠️ No holiday data found for {country_code}, using European default patterns with German Easter dates")

        # European school holiday approximations
        month = out[date_col].dt.month
        day = out[date_col].dt.day

        # Summer holidays (mid June to mid August)
        summer_holidays = ((month == 6) & (day >= 15)) | (month == 7) | ((month == 8) & (day <= 15))

        # Winter holidays (mid December to early January)
        winter_holidays = ((month == 12) & (day >= 20)) | ((month == 1) & (day <= 8))

        # Get Easter dates from Germany (since Easter is the same across Western Europe)
        easter_holidays_from_api = out['date_only'].apply(
            lambda x: is_vacation_day(x, 'germany', buffer_days) if x.month in [3, 4] else 0
        )

        # Autumn break (mid to late October)
        autumn_holidays = (month == 10) & (day >= 15) & (day <= 25)

        # Winter break (late February)
        winter_break = (month == 2) & (day >= 20)

        out['is_vacation_season'] = (summer_holidays | winter_holidays | easter_holidays_from_api |
                                   autumn_holidays | winter_break).astype(int)

    # Add specific holiday periods (common European patterns)
    out['is_summer_vacation'] = out[date_col].dt.month.isin([7, 8]).astype(int)
    out['is_winter_vacation'] = out[date_col].dt.month.isin([12, 1]).astype(int)
    out['is_easter_period'] = ((out[date_col].dt.month == 3) |
                               ((out[date_col].dt.month == 4) & (out[date_col].dt.day <= 15))).astype(int)

    # Remove temporary column
    out = out.drop('date_only', axis=1)

    vacation_count = out['is_vacation_season'].sum()
    total_count = len(out)
    print(f"✅ Added holiday features: {vacation_count}/{total_count} ({vacation_count/total_count*100:.1f}%) flights in vacation periods")

    return out


def build_flight_weather_features(
    df_flights: Optional[pd.DataFrame] = None,
    df_weather: Optional[pd.DataFrame] = None,
    df_airports: Optional[pd.DataFrame] = None,
    data_dir: Optional[Path] = None,
    save: bool = False,
    out_filename: str = "flight_weather_features.csv",
    add_time_features: bool = True,
    compute_relative_wind: bool = True,
    include_concurrency_features: bool = True,
    include_holiday_features: bool = True,
    holiday_country: str = 'norway',
) -> pd.DataFrame:
    """Produce merged flight+weather features.

    If DataFrames are not provided the function will attempt to load them from
    disk under `data_dir` (default: project `data/` folder).

    Returns the merged DataFrame and optionally writes it to
    `data_dir/out_filename`.
    """
    # Delegate to the pipeline which returns the computed artifacts.
    pipeline_outputs = flight_weather_pipeline(
        df_flights=df_flights,
        df_weather=df_weather,
        df_airports=df_airports,
        data_dir=data_dir,
        add_time_features=add_time_features,
        compute_relative_wind=compute_relative_wind,
        include_concurrency_features=include_concurrency_features,
        include_holiday_features=include_holiday_features,
        holiday_country=holiday_country,
    )

    merged = pipeline_outputs["merged"]

    if save:
        save_flight_weather_features(
            merged, data_dir=data_dir, out_filename=out_filename
        )

    return merged


def flight_weather_pipeline(
    df_flights: Optional[pd.DataFrame] = None,
    df_weather: Optional[pd.DataFrame] = None,
    df_airports: Optional[pd.DataFrame] = None,
    data_dir: Optional[Path] = None,
    add_time_features: bool = True,
    compute_relative_wind: bool = True,
    include_concurrency_features: bool = True,
    include_holiday_features: bool = True,
    holiday_country: str = 'norway',
) -> dict:
    """Run a deterministic pipeline that returns intermediate artifacts.

    Returns a dict with keys: 'processed_flights', 'cleaned_weather', 'weather_features', 'merged'.
    The function does NOT save to disk — use `save_flight_weather_features` to persist.
    """
    dp = DataPreprocessor()
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR

    # Load inputs if not provided
    if df_flights is None:
        flights_file = data_dir / "avinor_data/historical_flights.csv"
        if not flights_file.exists():
            raise FileNotFoundError(f"Flights file not found: {flights_file}")
        df_flights = pd.read_csv(flights_file)

    if df_weather is None:
        w_complete = data_dir / "weather_data_complete.csv"
        w_main = data_dir / "weather_data.csv"
        weather_file = w_complete if w_complete.exists() else w_main
        if not weather_file.exists():
            raise FileNotFoundError(
                f"Weather file not found (tried {w_complete} and {w_main})"
            )
        df_weather = pd.read_csv(weather_file)

    if df_airports is None:
        airports_file = data_dir / "airport_coordinates.csv"
        if airports_file.exists():
            df_airports = pd.read_csv(airports_file)
        else:
            df_airports = None

    # Process flights (use non-cancelled historical by default)
    processed_flights = dp.process_hist_flights(df_flights)

    # Clean and feature-engineer weather
    cleaned_weather = dp.process_weather(df_weather)
    weather_features = dp.feature_engineer_weather(cleaned_weather)

    # Merge weather onto flights
    merged = dp.merge_weather_with_flights(
        weather_features, processed_flights, df_airports=df_airports
    )

    # Add temporal cyclical features to the merged flight rows (optional)
    if add_time_features:
        merged = add_temporal_cyclical_features(merged, time_col="flight_datetime_hour")

    # Compute flight-relative headwind and crosswind (optional)
    if compute_relative_wind:
        merged = compute_flight_relative_wind(merged, df_airports=df_airports)

    # Add concurrency features (optional)
    if include_concurrency_features:
        merged = add_concurrency_features(
            merged, df_flights=processed_flights, df_airports=df_airports
        )

    # Add holiday/vacation features (optional)
    if include_holiday_features:
        merged = add_holiday_features(merged, country_code=holiday_country)

        # Add flight distance feature (in kilometers)
        merged = add_flight_distance_feature(merged, df_airports=df_airports)

    return {
        "processed_flights": processed_flights,
        "cleaned_weather": cleaned_weather,
        "weather_features": weather_features,
        "merged": merged,
    }
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth (in km)."""
    import math
    R = 6371.0  # Earth radius in kilometers
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan
    lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def add_flight_distance_feature(df, df_airports=None, dep_col_candidates=None, arr_col_candidates=None):
    """
    Adds 'flight_distance_km' column to df using airport coordinates from df_airports.
    """
    out = df.copy()
    dep_candidates = dep_col_candidates or ['dep_airport', 'origin', 'departure_airport', 'dep']
    arr_candidates = arr_col_candidates or ['arr_airport', 'destination', 'arrival_airport', 'arr']
    dep_col = next((c for c in dep_candidates if c in out.columns), None)
    arr_col = next((c for c in arr_candidates if c in out.columns), None)
    if df_airports is None or dep_col is None or arr_col is None:
        out['flight_distance_km'] = np.nan
        return out
    coords = df_airports.set_index('iata_code')[['latitude', 'longitude']].to_dict('index')
    def lookup_coords(code):
        if pd.isna(code):
            return (np.nan, np.nan)
        if code in coords:
            return (coords[code]['latitude'], coords[code]['longitude'])
        return (np.nan, np.nan)
    def compute_distance(row):
        lat1, lon1 = lookup_coords(row[dep_col])
        lat2, lon2 = lookup_coords(row[arr_col])
        return haversine_distance(lat1, lon1, lat2, lon2)
    out['flight_distance_km'] = out.apply(compute_distance, axis=1)
    return out


def compute_flight_relative_wind(
    df: pd.DataFrame,
    df_airports: Optional[pd.DataFrame] = None,
    origin_col_candidates=None,
    dest_col_candidates=None,
    wind_speed_col: str = "wind_speed",
    wind_dir_col: str = "wind_direction",
) -> pd.DataFrame:
    """Compute flight-relative headwind and crosswind.

    - Attempts to find origin/destination airport columns among common names
    - Uses `df_airports` (expects `iata_code`, `latitude`, `longitude`) to get coords
    - Adds columns: `flight_bearing_deg`, `wind_headwind`, `wind_crosswind`
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    origin_candidates = origin_col_candidates or [
        "origin",
        "dep_airport",
        "departure_airport",
        "dep",
    ]
    dest_candidates = dest_col_candidates or [
        "destination",
        "arr_airport",
        "arrival_airport",
        "arr",
    ]

    origin_col = next((c for c in origin_candidates if c in out.columns), None)
    dest_col = next((c for c in dest_candidates if c in out.columns), None)

    # Helper: bearing calculation
    import math

    def _bearing(lat1, lon1, lat2, lon2):
        try:
            lat1 = math.radians(lat1)
            lat2 = math.radians(lat2)
            diff_long = math.radians(lon2 - lon1)
            x = math.sin(diff_long) * math.cos(lat2)
            y = math.cos(lat1) * math.sin(lat2) - (
                math.sin(lat1) * math.cos(lat2) * math.cos(diff_long)
            )
            initial_bearing = math.degrees(math.atan2(x, y))
            return (initial_bearing + 360) % 360
        except Exception:
            return np.nan

    # If coords already present in dataframe, use them
    if {"origin_lat", "origin_lon", "dest_lat", "dest_lon"}.issubset(out.columns):
        out["flight_bearing_deg"] = out.apply(
            lambda r: (
                _bearing(r["origin_lat"], r["origin_lon"], r["dest_lat"], r["dest_lon"])
                if pd.notna(r["origin_lat"]) and pd.notna(r["dest_lat"])
                else np.nan
            ),
            axis=1,
        )
    elif df_airports is not None and origin_col and dest_col:
        # build lookup
        coords = df_airports.set_index("iata_code")[["latitude", "longitude"]].to_dict(
            "index"
        )

        def lookup_coords(code):
            if pd.isna(code):
                return (np.nan, np.nan)
            if code in coords:
                return (coords[code]["latitude"], coords[code]["longitude"])
            return (np.nan, np.nan)

        out[["origin_lat", "origin_lon"]] = out[origin_col].apply(
            lambda c: pd.Series(lookup_coords(c))
        )
        out[["dest_lat", "dest_lon"]] = out[dest_col].apply(
            lambda c: pd.Series(lookup_coords(c))
        )

        out["flight_bearing_deg"] = out.apply(
            lambda r: (
                _bearing(r["origin_lat"], r["origin_lon"], r["dest_lat"], r["dest_lon"])
                if pd.notna(r["origin_lat"]) and pd.notna(r["dest_lat"])
                else np.nan
            ),
            axis=1,
        )
    else:
        # cannot compute bearing
        out["flight_bearing_deg"] = np.nan

    # Project wind onto flight bearing
    def project_head_cross(ws, wd, bearing):
        try:
            if pd.isna(ws) or pd.isna(wd) or pd.isna(bearing):
                return (np.nan, np.nan)
            # theta = angle between wind vector (from direction wd) and flight track
            theta = math.radians((wd - bearing + 360) % 360)
            head = ws * math.cos(theta)
            cross = ws * math.sin(theta)
            return (head, cross)
        except Exception:
            return (np.nan, np.nan)

    # compute head and cross
    comp = out.apply(
        lambda r: project_head_cross(
            r.get(wind_speed_col), r.get(wind_dir_col), r.get("flight_bearing_deg")
        ),
        axis=1,
    )
    comp_df = pd.DataFrame(
        comp.tolist(), index=out.index, columns=["wind_headwind", "wind_crosswind"]
    )
    out = pd.concat([out, comp_df], axis=1)

    return out


def add_temporal_cyclical_features(
    df: pd.DataFrame,
    time_col: str = "flight_datetime_hour",
    keep_datetime: bool = True,
) -> pd.DataFrame:
    """Add temporal features and cyclical encodings based on `time_col`.

    - Keeps the original datetime column (does not drop it) so concatenation/joins
      remain simple.
    - Adds: hour, dayofweek, month, dayofyear, is_weekend
    - Adds cyclical encodings: hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, doy_sin, doy_cos

    Parameters
    - df: DataFrame to operate on (a copy is returned)
    - time_col: column name with datetimes; if missing, function will try 'datetime'
    - keep_datetime: if True (default) the datetime column remains; function never removes other columns.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    col = (
        time_col
        if time_col in out.columns
        else ("datetime" if "datetime" in out.columns else None)
    )
    if col is None:
        # nothing to do
        return out

    out[col] = pd.to_datetime(out[col], utc=True)
    out[col] = out[col].dt.floor("h")

    out["hour"] = out[col].dt.hour
    out["dayofweek"] = out[col].dt.dayofweek
    out["month"] = out[col].dt.month
    out["dayofyear"] = out[col].dt.dayofyear
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)

    # cyclical encodings
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)

    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12)

    out["doy_sin"] = np.sin(2 * np.pi * (out["dayofyear"] - 1) / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * (out["dayofyear"] - 1) / 365.25)

    return out


def add_concurrency_features(
    df: pd.DataFrame,
    df_flights: Optional[pd.DataFrame] = None,
    df_airports: Optional[pd.DataFrame] = None,
    time_col: str = "flight_datetime_hour",
    origin_col: str = "dep_airport",
    dest_col: str = "arr_airport",
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Add concurrency features using cached lookup dataset for speed.

    First tries to load pre-computed concurrency data. If not found,
    falls back to computing on-the-fly.

    Adds columns:
    - origin_departures_hour, origin_arrivals_hour (concurrency at departure airport)
    - dest_departures_hour, dest_arrivals_hour (concurrency at destination airport)
    - origin_group_departures_hour, origin_group_arrivals_hour (by airport group if available)
    - dest_group_departures_hour, dest_group_arrivals_hour
    """
    from src.traffic import load_concurrency_lookup, create_concurrency_lookup_dataset
    


    if df is None or df.empty:
        return df

    out = df.copy()

    # Ensure time column exists and is datetime
    if time_col not in out.columns:
        return out

    out[time_col] = pd.to_datetime(out[time_col], utc=True).dt.floor("h")
    data_dir = data_dir or DATA_DIR

    # Try to load cached concurrency lookup
    concurrency_df = load_concurrency_lookup(data_dir)

    if concurrency_df is None:
        print("⚠️ No cached concurrency data found. Computing on-the-fly...")
        if df_flights is None:
            print(
                "⚠️ Cannot compute concurrency: no flight data provided. Skipping concurrency features."
            )
            return out

        # Fallback: create dataset on-the-fly
        try:
            create_concurrency_lookup_dataset(
                df_flights, df_airports, data_dir=data_dir, by_groups=True
            )
            concurrency_df = load_concurrency_lookup(data_dir)
        except Exception as e:
            print(f"⚠️ Failed to compute concurrency: {e}")
            return out

    if concurrency_df is None or concurrency_df.empty:
        print("⚠️ No concurrency data available. Skipping concurrency features.")
        return out

    # Prepare lookup data
    concurrency_df["datetime"] = pd.to_datetime(
        concurrency_df["datetime"], utc=True
    ).dt.floor("h")

    # Determine if we have airport-level or group-level data
    has_airport_data = (
        "airport" in concurrency_df.columns and concurrency_df["airport"].notna().any()
    )
    has_group_data = (
        "airport_group" in concurrency_df.columns
        and concurrency_df["airport_group"].notna().any()
    )

    # Add airport-level concurrency features
    if has_airport_data:
        airport_lookup = concurrency_df[concurrency_df["airport"].notna()].copy()

        # Create efficient lookup dictionaries
        dep_lookup = airport_lookup.set_index(["datetime", "airport"])[
            "dep_concurrency"
        ].to_dict()
        arr_lookup = airport_lookup.set_index(["datetime", "airport"])[
            "arr_concurrency"
        ].to_dict()

        # Add origin airport concurrency
        if origin_col in out.columns:
            out["origin_departures_hour"] = out.apply(
                lambda row: (
                    dep_lookup.get((row[time_col], row[origin_col]), 0)
                    if pd.notna(row.get(time_col)) and pd.notna(row.get(origin_col))
                    else 0
                ),
                axis=1,
            )
            out["origin_arrivals_hour"] = out.apply(
                lambda row: (
                    arr_lookup.get((row[time_col], row[origin_col]), 0)
                    if pd.notna(row.get(time_col)) and pd.notna(row.get(origin_col))
                    else 0
                ),
                axis=1,
            )

        # Add destination airport concurrency
        if dest_col in out.columns:
            out["dest_departures_hour"] = out.apply(
                lambda row: (
                    dep_lookup.get((row[time_col], row.get(dest_col)), 0)
                    if pd.notna(row.get(time_col)) and pd.notna(row.get(dest_col))
                    else 0
                ),
                axis=1,
            )
            out["dest_arrivals_hour"] = out.apply(
                lambda row: (
                    arr_lookup.get((row[time_col], row.get(dest_col)), 0)
                    if pd.notna(row.get(time_col)) and pd.notna(row.get(dest_col))
                    else 0
                ),
                axis=1,
            )

    # Add group-level concurrency features
    if has_group_data:
        group_lookup = concurrency_df[concurrency_df["airport_group"].notna()].copy()

        # Create efficient lookup dictionaries for groups
        group_dep_lookup = group_lookup.set_index(["datetime", "airport_group"])[
            "dep_concurrency"
        ].to_dict()
        group_arr_lookup = group_lookup.set_index(["datetime", "airport_group"])[
            "arr_concurrency"
        ].to_dict()

        # Add origin group concurrency
        origin_group_col = "dep_airport_group"
        if origin_group_col in out.columns:
            out["origin_group_departures_hour"] = out.apply(
                lambda row: (
                    group_dep_lookup.get((row[time_col], row.get(origin_group_col)), 0)
                    if pd.notna(row.get(time_col))
                    and pd.notna(row.get(origin_group_col))
                    else 0
                ),
                axis=1,
            )
            out["origin_group_arrivals_hour"] = out.apply(
                lambda row: (
                    group_arr_lookup.get((row[time_col], row.get(origin_group_col)), 0)
                    if pd.notna(row.get(time_col))
                    and pd.notna(row.get(origin_group_col))
                    else 0
                ),
                axis=1,
            )

        # Add destination group concurrency
        dest_group_col = "arr_airport_group"
        if dest_group_col in out.columns:
            out["dest_group_departures_hour"] = out.apply(
                lambda row: (
                    group_dep_lookup.get((row[time_col], row.get(dest_group_col)), 0)
                    if pd.notna(row.get(time_col)) and pd.notna(row.get(dest_group_col))
                    else 0
                ),
                axis=1,
            )
            out["dest_group_arrivals_hour"] = out.apply(
                lambda row: (
                    group_arr_lookup.get((row[time_col], row.get(dest_group_col)), 0)
                    if pd.notna(row.get(time_col)) and pd.notna(row.get(dest_group_col))
                    else 0
                ),
                axis=1,
            )

    # Fill any remaining NaN values with 0
    concurrency_cols = [
        col
        for col in out.columns
        if "concurrency" in col or "departures_hour" in col or "arrivals_hour" in col
    ]
    for col in concurrency_cols:
        out[col] = out[col].fillna(0)

    return out


def save_flight_weather_features(
    df: pd.DataFrame,
    data_dir: Optional[Path] = None,
    out_filename: str = "flight_weather_features.csv",
) -> Path:
    """Persist the merged features DataFrame to CSV and return the path."""
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    out_file = Path(data_dir) / out_filename
    df.to_csv(out_file, index=False)
    print(f"✅ Saved flight+weather features to: {out_file}")
    return out_file


if __name__ == "__main__":
    # simple CLI when executed directly
    try:
        df = build_flight_weather_features()
        print(f"✅ Built features: {len(df):,} rows, columns: {list(df.columns)}")
    except Exception as exc:
        print(f"❌ Feature building failed: {exc}")
