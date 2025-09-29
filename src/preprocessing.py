import os
import math
import numpy as np
import pandas as pd
from pandas import DataFrame


class DataPreprocessor:
    def __init__(self):
        pass

    def process_hist_flights(self, df_hist: DataFrame) -> DataFrame:
        """General processing of historical flight data"""
        df_processed_hist = df_hist.copy()

        # Add cyclical temporal features for std, sta, atd, ata
        def add_time_sin_cos(df, col):
            if col in df.columns:
                dt = pd.to_datetime(df[col], errors='coerce')
                df[f'{col}_hour'] = dt.dt.hour
                df[f'{col}_minute'] = dt.dt.minute
                time_float = df[f'{col}_hour'] + df[f'{col}_minute'] / 60.0
                df[f'{col}_sin'] = np.sin(2 * np.pi * time_float / 24)
                df[f'{col}_cos'] = np.cos(2 * np.pi * time_float / 24)
            return df

        for time_col in ['std', 'sta', 'atd', 'ata']:
            df_processed_hist = add_time_sin_cos(df_processed_hist, time_col)

        # Fill missing airport groups with "OUT"
        df_processed_hist["arr_airport_group"] = df_processed_hist[
            "arr_airport_group"
        ].fillna("OUT")
        df_processed_hist["dep_airport_group"] = df_processed_hist[
            "dep_airport_group"
        ].fillna("OUT")

        # Fill missing service_type values within each flight_id group and keep original format
        df_processed_hist["service_type"] = df_processed_hist["service_type"].fillna(
            "X"
        )

        # Drop rows where cancelled is 1
        df_processed_hist = df_processed_hist[df_processed_hist["cancelled"] == 0]

        std_parsed = (
            pd.to_datetime(df_processed_hist["std"])
            if "std" in df_processed_hist.columns
            else None
        )
        sta_parsed = (
            pd.to_datetime(df_processed_hist["sta"])
            if "sta" in df_processed_hist.columns
            else None
        )

        df_processed_hist["std"] = std_parsed
        df_processed_hist["sta"] = sta_parsed

        atd_parsed = (
            pd.to_datetime(df_processed_hist["atd"])
            if "atd" in df_processed_hist.columns
            else None
        )
        ata_parsed = (
            pd.to_datetime(df_processed_hist["ata"])
            if "ata" in df_processed_hist.columns
            else None
        )

        df_processed_hist["atd"] = atd_parsed
        df_processed_hist["ata"] = ata_parsed

        # remove cancelled column
        df_processed_hist = df_processed_hist.drop(columns=["cancelled"])
        print("Missing values after processing:")
        print(df_processed_hist.isna().sum())

        print("dropping rows with missing actual times")
        df_processed_hist = df_processed_hist.dropna(subset=["atd", "ata"])
        print("Final missing values:")
        print(df_processed_hist.isna().sum())
        return df_processed_hist

    def process_hist_flights_cancelled(self, df_hist: DataFrame) -> DataFrame:
        """General processing of historical flight data and removal of unused columns"""
        # Historical Flights
        # todo: missing atd and ata:
        # train a model without these values, predict these values, and train a model with these values
        # or completely drop these rows

        df_processed_hist = df_hist.copy()

        std_parsed = (
            pd.to_datetime(df_processed_hist["std"])
            if "std" in df_processed_hist.columns
            else None
        )
        sta_parsed = (
            pd.to_datetime(df_processed_hist["sta"])
            if "sta" in df_processed_hist.columns
            else None
        )

        df_processed_hist["std"] = std_parsed
        df_processed_hist["sta"] = sta_parsed

        # Sort by std (scheduled time of departure)
        df_processed_hist = df_processed_hist.sort_values("std")

        # set cancelled to 1 if atd and ata are missing
        df_processed_hist[
            df_processed_hist["cancelled"].isna()
            | df_processed_hist["ata"].isna() & df_processed_hist["atd"].isna()
        ]["cancelled"] = 1

        # drop actual times for cancellation prediction
        df_processed_hist = df_processed_hist.drop(columns=["atd", "ata"])

        # Fill missing
        df_processed_hist["arrival_airport_group"] = df_processed_hist[
            "arrival_airport_group"
        ].fillna("OUT")
        df_processed_hist["departure_airport_group"] = df_processed_hist[
            "departure_airport_group"
        ].fillna("OUT")

        print("Missing values after processing:")
        print(df_processed_hist.isna().sum())

        return df_processed_hist

    def process_future_flights(self, df_fut: DataFrame) -> DataFrame:
        # Implement data cleaning logic here
        df_processed_fut = df_fut.copy()
        return df_processed_fut

    # ------------------------
    # Weather preprocessing
    # ------------------------
    def process_weather(self, df_weather: DataFrame) -> DataFrame:
        """Clean and standardize raw weather dataframe.

        - Normalize common column names (temperature_2m -> temperature, etc.)
        - Parse timestamps to UTC and floor to the hour
        - Keep a sensible set of columns
        - Drop duplicates and sort
        - Interpolate small gaps (per-airport)

        Returns a cleaned DataFrame with at least columns:
        ['airport', 'datetime', 'temperature', 'humidity', 'precipitation',
         'pressure', 'cloud_cover', 'wind_speed', 'wind_direction', 'visibility',
         'latitude', 'longitude'] (columns may be absent if not in source)
        """
        if df_weather is None or df_weather.empty:
            return df_weather.copy()

        df = df_weather.copy()

        # Common renames from Open-Meteo and other sources
        rename_map = {
            "time": "datetime",
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "precipitation": "precipitation",
            "surface_pressure": "pressure",
            "cloud_cover": "cloud_cover",
            "wind_speed_10m": "wind_speed",
            "wind_direction_10m": "wind_direction",
            "visibility": "visibility",
            "airport_code": "airport",
            "iata_code": "airport",
        }

        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Ensure datetime is parsed
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
            # also add a floored hour column for easy joins
            df["datetime"] = df["datetime"].dt.tz_convert("UTC")
        else:
            # try to infer from common names
            raise ValueError(
                "Input weather DataFrame must contain a datetime/time column"
            )

        # Floor to the hour (we treat hourly weather)
        df["datetime"] = df["datetime"].dt.floor("h")

        # Standard column set we keep if present
        keep_cols = [
            "airport",
            "datetime",
            "temperature",
            "humidity",
            "precipitation",
            "pressure",
            "cloud_cover",
            "wind_speed",
            "wind_direction",
            "visibility",
            "latitude",
            "longitude",
        ]

        existing_keep = [c for c in keep_cols if c in df.columns]
        df = df[existing_keep].copy()

        # Drop exact duplicates
        df = df.drop_duplicates()

        # Sort and reset index
        df = df.sort_values(["airport", "datetime"]).reset_index(drop=True)

        # Apply intelligent weather trend-based NaN filling
        df = self._fill_weather_gaps_with_trends(df)

        return df

    def _fill_weather_gaps_with_trends(self, df: DataFrame) -> DataFrame:
        """Fill weather NaN values using intelligent weather patterns and trends.

        Uses realistic weather patterns instead of generic fallbacks:
        - Temperature: diurnal cycles, seasonal patterns, nearby airport correlation
        - Precipitation: realistic probability-based filling
        - Wind: prevailing wind patterns and seasonal variations
        - Pressure: smooth atmospheric patterns
        - Humidity: temperature-dependent realistic ranges
        """
        if df.empty:
            return df

        df = df.copy()

        # Process each airport separately to maintain spatial consistency
        processed_groups = []
        for airport, airport_df in df.groupby("airport"):
            filled_df = self._fill_airport_weather_trends(airport_df, airport)
            processed_groups.append(filled_df)

        result = pd.concat(processed_groups, ignore_index=True)
        result = result.sort_values(["airport", "datetime"]).reset_index(drop=True)

        return result

    def _fill_airport_weather_trends(self, df: DataFrame, airport: str) -> DataFrame:
        """Fill NaN values for a single airport using realistic weather trends"""
        df = df.copy().sort_values("datetime")

        if df.empty:
            return df

        # Use airport parameter for potential future enhancements (e.g., location-specific patterns)
        _ = airport  # Acknowledge parameter

        # Add temporal features for pattern recognition
        df["hour"] = df["datetime"].dt.hour
        df["day_of_year"] = df["datetime"].dt.dayofyear
        df["month"] = df["datetime"].dt.month
        df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)
        df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)

        # Temperature filling with diurnal and seasonal patterns
        df = self._fill_temperature_with_patterns(df)

        # Precipitation filling with realistic probability
        df = self._fill_precipitation_realistically(df)

        # Wind filling with prevailing patterns
        df = self._fill_wind_with_patterns(df)

        # Pressure filling with smooth atmospheric patterns
        df = self._fill_pressure_smoothly(df)

        # Humidity filling based on temperature relationship
        df = self._fill_humidity_temp_dependent(df)

        # Other numeric columns with interpolation and seasonal patterns
        other_numeric = ["cloud_cover", "visibility"]
        for col in other_numeric:
            if col in df.columns:
                df = self._fill_generic_weather_var(df, col)

        # Clean up temporary columns
        df = df.drop(
            ["hour", "day_of_year", "month", "is_winter", "is_summer"],
            axis=1,
            errors="ignore",
        )

        return df

    def _fill_temperature_with_patterns(self, df: DataFrame) -> DataFrame:
        """Fill temperature using realistic diurnal and seasonal patterns"""
        if "temperature" not in df.columns:
            return df

        df = df.copy()
        temp_col = "temperature"

        # First, basic interpolation for small gaps
        df[temp_col] = df[temp_col].interpolate(method="linear", limit=6)

        # For remaining NaNs, use pattern-based filling
        if df[temp_col].isna().any():

            # Calculate seasonal temperature baseline for this location
            valid_temps = df[temp_col].dropna()
            if len(valid_temps) > 24:  # Need at least a day of data

                # Compute typical diurnal pattern (temperature by hour)
                hourly_pattern = df.groupby("hour")[temp_col].mean()

                # Compute seasonal pattern (temperature by day of year)
                daily_pattern = df.groupby("day_of_year")[temp_col].mean()

                # Fill NaN values using combined patterns
                for idx, row in df[df[temp_col].isna()].iterrows():
                    hour = row["hour"]
                    day_of_year = row["day_of_year"]

                    # Base temperature from available data
                    base_temp = valid_temps.mean()

                    # Adjust for time of day (diurnal cycle)
                    if hour in hourly_pattern.index:
                        hourly_adj = hourly_pattern[hour] - base_temp
                    else:
                        # Simulate typical diurnal pattern: cooler at night, warmer in afternoon
                        typical_diurnal = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
                        hourly_adj = typical_diurnal

                    # Adjust for season (annual cycle)
                    if day_of_year in daily_pattern.index:
                        seasonal_adj = daily_pattern[day_of_year] - base_temp
                    else:
                        # Simulate seasonal pattern: cooler in winter, warmer in summer
                        seasonal_adj = 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

                    # Combine adjustments with some random variation
                    filled_temp = (
                        base_temp
                        + 0.7 * hourly_adj
                        + 0.3 * seasonal_adj
                        + np.random.normal(0, 1)
                    )
                    df.loc[idx, temp_col] = filled_temp

            else:
                # Fallback for insufficient data: use reasonable defaults with variation
                mean_temp = (
                    valid_temps.mean() if len(valid_temps) > 0 else 15.0
                )  # 15°C reasonable default
                for idx, row in df[df[temp_col].isna()].iterrows():
                    # Add diurnal variation: cooler at night, warmer during day
                    diurnal_var = 8 * np.sin(2 * np.pi * (row["hour"] - 6) / 24)
                    # Add seasonal variation
                    seasonal_var = 15 * np.sin(
                        2 * np.pi * (row["day_of_year"] - 80) / 365
                    )
                    # Add random noise
                    noise = np.random.normal(0, 2)

                    filled_temp = mean_temp + diurnal_var + seasonal_var + noise
                    df.loc[idx, temp_col] = filled_temp

        return df

    def _fill_precipitation_realistically(self, df: DataFrame) -> DataFrame:
        """Fill precipitation with realistic probability-based patterns"""
        if "precipitation" not in df.columns:
            return df

        df = df.copy()
        precip_col = "precipitation"

        # Most missing precipitation should be 0 (no rain is more common)
        # But occasionally add realistic rain events

        valid_precip = df[precip_col].dropna()

        for idx, row in df[df[precip_col].isna()].iterrows():
            # 80% chance of no precipitation (realistic for most locations/times)
            if np.random.random() < 0.8:
                df.loc[idx, precip_col] = 0.0
            else:
                # When it does rain, use patterns from existing data
                if len(valid_precip[valid_precip > 0]) > 0:
                    # Sample from existing non-zero precipitation values
                    rain_values = valid_precip[valid_precip > 0]
                    df.loc[idx, precip_col] = np.random.choice(rain_values)
                else:
                    # Fallback: light rain (1-5mm typical for light rain)
                    df.loc[idx, precip_col] = np.random.uniform(0.5, 3.0)

        return df

    def _fill_wind_with_patterns(self, df: DataFrame) -> DataFrame:
        """Fill wind speed and direction with prevailing patterns"""
        df = df.copy()

        # Fill wind speed
        if "wind_speed" in df.columns:
            valid_wind_speed = df["wind_speed"].dropna()

            for idx, row in df[df["wind_speed"].isna()].iterrows():
                if len(valid_wind_speed) > 0:
                    # Use seasonal patterns: typically windier in winter/spring
                    base_wind = valid_wind_speed.median()
                    seasonal_factor = (
                        1.2 if row["is_winter"] else 0.9 if row["is_summer"] else 1.0
                    )
                    # Add some variation
                    filled_wind = base_wind * seasonal_factor + np.random.normal(0, 1)
                    df.loc[idx, "wind_speed"] = max(
                        0, filled_wind
                    )  # Wind speed can't be negative
                else:
                    # Reasonable default wind speed with variation
                    df.loc[idx, "wind_speed"] = max(
                        0, np.random.normal(8, 3)
                    )  # 8 m/s ± 3 m/s

        # Fill wind direction
        if "wind_direction" in df.columns:
            valid_wind_dir = df["wind_direction"].dropna()

            for idx, row in df[df["wind_direction"].isna()].iterrows():
                if len(valid_wind_dir) > 0:
                    # Find prevailing wind direction (most common direction ± 30°)
                    hist, bin_edges = np.histogram(valid_wind_dir, bins=36)  # 10° bins
                    prevailing_bin = np.argmax(hist)
                    prevailing_dir = (
                        bin_edges[prevailing_bin] + bin_edges[prevailing_bin + 1]
                    ) / 2

                    # Add variation around prevailing direction
                    filled_dir = (prevailing_dir + np.random.normal(0, 30)) % 360
                    df.loc[idx, "wind_direction"] = filled_dir
                else:
                    # Random direction if no data available
                    df.loc[idx, "wind_direction"] = np.random.uniform(0, 360)

        return df

    def _fill_pressure_smoothly(self, df: DataFrame) -> DataFrame:
        """Fill atmospheric pressure with smooth interpolation"""
        if "pressure" not in df.columns:
            return df

        df = df.copy()

        # Pressure changes are typically gradual
        df["pressure"] = df["pressure"].interpolate(
            method="linear", limit=12
        )  # Allow longer interpolation

        # For remaining NaNs, use standard atmospheric pressure with small variations
        for idx, row in df[df["pressure"].isna()].iterrows():
            # Standard atmospheric pressure is 1013.25 hPa
            # Typical variations are ±30 hPa
            filled_pressure = 1013.25 + np.random.normal(0, 15)
            df.loc[idx, "pressure"] = filled_pressure

        return df

    def _fill_humidity_temp_dependent(self, df: DataFrame) -> DataFrame:
        """Fill humidity based on realistic temperature relationships"""
        if "humidity" not in df.columns or "temperature" not in df.columns:
            return df

        df = df.copy()

        for idx, row in df[df["humidity"].isna()].iterrows():
            temp = row["temperature"]

            if pd.notna(temp):
                # Realistic humidity based on temperature
                # Colder air: typically higher relative humidity
                # Warmer air: more variable humidity
                if temp < 0:  # Very cold
                    humidity = np.random.uniform(70, 90)
                elif temp < 10:  # Cold
                    humidity = np.random.uniform(60, 85)
                elif temp < 20:  # Moderate
                    humidity = np.random.uniform(50, 80)
                elif temp < 30:  # Warm
                    humidity = np.random.uniform(40, 75)
                else:  # Hot
                    humidity = np.random.uniform(30, 70)
            else:
                # Fallback if temperature is also missing
                humidity = np.random.uniform(50, 80)  # Typical range

            df.loc[idx, "humidity"] = humidity

        return df

    def _fill_generic_weather_var(self, df: DataFrame, col: str) -> DataFrame:
        """Fill other weather variables with interpolation and reasonable defaults"""
        if col not in df.columns:
            return df

        df = df.copy()

        # First try interpolation
        df[col] = df[col].interpolate(method="linear", limit=6)

        # Fill remaining with defaults based on variable type
        defaults = {
            "cloud_cover": lambda: np.random.uniform(20, 80),  # Variable cloud cover
            "visibility": lambda: np.random.uniform(
                8000, 15000
            ),  # Good visibility typical
        }

        if col in defaults:
            for idx in df[df[col].isna()].index:
                df.loc[idx, col] = defaults[col]()

        return df

    def feature_engineer_weather(self, df_weather: DataFrame) -> DataFrame:
        """Create useful features from cleaned hourly weather data.

        Adds:
        - temporal features: hour, dayofweek, month, is_weekend
        - cyclical transforms: hour_sin, hour_cos, dow_sin, dow_cos
        - short-term trends: temp_diff_1h, temp_roll_3h
        - wind vector components: wind_u, wind_v

        Input expects the dataframe returned from `process_weather` with columns
        including 'datetime' (datetime64[ns,UTC]) and 'wind_speed' & 'wind_direction'.
        """
        if df_weather is None or df_weather.empty:
            return df_weather.copy()

        df = df_weather.copy()

        if "datetime" not in df.columns:
            raise ValueError("weather DataFrame must contain a `datetime` column")

        # Ensure datetime is timezone-aware and floored to the hour
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.floor("h")

        # Temporal features
        df["hour"] = df["datetime"].dt.hour
        df["dayofweek"] = df["datetime"].dt.dayofweek
        df["month"] = df["datetime"].dt.month
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        # Cyclical encodings
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

        # Short-term trends and rolling stats per airport (with NaN handling)
        df = df.sort_values(["airport", "datetime"])

        # Temperature difference (fill NaN with 0 = no change)
        df["temp_diff_1h"] = df.groupby("airport")["temperature"].diff().fillna(0.0)

        # Rolling temperature (always has min_periods=1, fill any remaining NaN)
        df["temp_roll_3h"] = (
            df.groupby("airport")["temperature"]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df["temp_roll_3h"] = df["temp_roll_3h"].fillna(
            df["temperature"]
        )  # Fill with current temp if needed

        # Rolling precipitation (sum, fill NaN with 0)
        df["precip_roll_3h"] = (
            df.groupby("airport")["precipitation"]
            .rolling(window=3, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        df["precip_roll_3h"] = df["precip_roll_3h"].fillna(0.0)

        # Wind vector components: convert degrees to radians; note convention: wind_direction is meteorological (degrees from north)
        # We compute u (east-west) and v (north-south) components where positive u = eastward, positive v = northward.
        if "wind_speed" in df.columns and "wind_direction" in df.columns:
            # meteorological wind_direction: degrees from north, clockwise. To convert to mathematical angle from east (for u/v):
            # theta_math = 90 - wind_dir
            theta = np.deg2rad(90 - df["wind_direction"].fillna(0))
            df["wind_u"] = df["wind_speed"].fillna(0) * np.cos(theta)
            df["wind_v"] = df["wind_speed"].fillna(0) * np.sin(theta)
        else:
            df["wind_u"] = np.nan
            df["wind_v"] = np.nan

        return df

    def merge_weather_with_flights(
        self,
        df_weather: DataFrame,
        df_flights: DataFrame,
        df_airports: DataFrame = None,
        flight_time_col: str = "std",
        flight_origin_col: str = "origin",
        flight_dest_col: str = "destination",
    ) -> DataFrame:
        """Attach weather to flights and optionally compute headwind/crosswind components.

        - Merges weather onto flights by matching airport (origin) and floored hour of `flight_time_col`.
        - If `df_airports` is provided and `flight_dest_col` exists, computes bearing from origin->destination and
          then headwind and crosswind components relative to that bearing.

        NOTE: This function is permissive about column names but will raise informative errors if required columns
        are missing.
        """
        if df_flights is None or df_flights.empty:
            return df_flights.copy()

        flights = df_flights.copy()
        weather = df_weather.copy()

        # Ensure time columns
        if flight_time_col not in flights.columns:
            raise ValueError(
                f"Flights dataframe must contain time column `{flight_time_col}`"
            )

        # Floor times to hours for joining
        flights["flight_datetime_hour"] = pd.to_datetime(
            flights[flight_time_col], errors="coerce", utc=True
        ).dt.floor("h")
        weather["datetime"] = pd.to_datetime(weather["datetime"], utc=True).dt.floor(
            "h"
        )

        # Determine origin column name
        if flight_origin_col not in flights.columns:
            # Try common alternatives
            alt = None
            for candidate in [
                "origin",
                "origin_airport",
                "dep_airport",
                "departure_airport",
                "airport",
            ]:
                if candidate in flights.columns:
                    alt = candidate
                    break
            if alt is None:
                raise ValueError(
                    "Could not find an origin airport column in flights dataframe"
                )
            flight_origin_col = alt

        # Merge weather (left join)
        merged = pd.merge(
            flights,
            weather,
            left_on=[flight_origin_col, "flight_datetime_hour"],
            right_on=["airport", "datetime"],
            how="left",
            suffixes=("", "_weather"),
        )

        # If airport coordinates and destination are available, compute bearing and wind components relative to flight
        if df_airports is not None and flight_dest_col in merged.columns:
            # Build coords lookup
            coords = df_airports.set_index("iata_code")[
                ["latitude", "longitude"]
            ].to_dict("index")

            def get_coords(code):
                if pd.isna(code):
                    return (np.nan, np.nan)
                if code in coords:
                    return (coords[code]["latitude"], coords[code]["longitude"])
                return (np.nan, np.nan)

            # origin coords
            merged[["origin_lat", "origin_lon"]] = merged[flight_origin_col].apply(
                lambda c: pd.Series(get_coords(c))
            )
            merged[["dest_lat", "dest_lon"]] = merged[flight_dest_col].apply(
                lambda c: pd.Series(get_coords(c))
            )

            # compute bearing origin->dest
            merged["flight_bearing_deg"] = merged.apply(
                lambda r: (
                    self._calculate_bearing(
                        r["origin_lat"], r["origin_lon"], r["dest_lat"], r["dest_lon"]
                    )
                    if pd.notna(r["origin_lat"]) and pd.notna(r["dest_lat"])
                    else np.nan
                ),
                axis=1,
            )

            # compute headwind & crosswind if wind available
            if "wind_speed" in merged.columns and "wind_direction" in merged.columns:
                # difference between wind direction (meteo) and flight bearing
                # wind_dir and flight_bearing are degrees; convert to radians
                def comp_head_cross(ws, wd, bearing):
                    if pd.isna(ws) or pd.isna(wd) or pd.isna(bearing):
                        return (np.nan, np.nan)
                    # relative angle: positive when wind is from ahead (headwind positive)
                    # compute angle difference: wind comes from wd degrees (from north clockwise)
                    # headwind = ws * cos(theta) where theta = angle between wind vector and aircraft ground-track
                    theta = math.radians((wd - bearing + 360) % 360)
                    head = ws * math.cos(theta)
                    cross = ws * math.sin(theta)
                    return (head, cross)

                comp = merged.apply(
                    lambda r: comp_head_cross(
                        r["wind_speed"], r["wind_direction"], r["flight_bearing_deg"]
                    ),
                    axis=1,
                )
                comp_df = pd.DataFrame(
                    comp.tolist(),
                    index=merged.index,
                    columns=["wind_headwind", "wind_crosswind"],
                )
                merged = pd.concat([merged, comp_df], axis=1)

        return merged

    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        """Calculate bearing in degrees from point1 to point2 using haversine formulas.
        Returns bearing in degrees (0..360) where 0 = north, 90 = east.
        """
        try:
            lat1 = math.radians(lat1)
            lat2 = math.radians(lat2)
            diff_long = math.radians(lon2 - lon1)

            x = math.sin(diff_long) * math.cos(lat2)
            y = math.cos(lat1) * math.sin(lat2) - (
                math.sin(lat1) * math.cos(lat2) * math.cos(diff_long)
            )

            initial_bearing = math.degrees(math.atan2(x, y))
            compass_bearing = (initial_bearing + 360) % 360

            return compass_bearing
        except Exception:
            return np.nan


if __name__ == "__main__":
    df_hist_original = pd.read_csv(
        os.path.abspath("data/avinor_data/historical_flights.csv")
    )
    df_hist = df_hist_original.copy()
    preprocessor = DataPreprocessor()
    df_processed_hist = preprocessor.process_hist_flights_cancelled(df_hist)
    print(df_processed_hist.head())
