"""Concurrency utilities: compute arrivals/departures per airport and coverage diagnostics.

Main functions:
- compute_concurrency_with_coverage(...) -> (counts_time, counts_day, coverage_df)
- save_concurrency_tables(...)

Design notes:
- Uses scheduled times only: `std` for departures and `sta` for arrivals.
- Builds hourly or daily counts and always includes a master airport list (from `df_airports`
  or inferred from flights) so outputs contain zeros instead of NaNs.
- Returns a per-airport coverage summary so you can filter out airports with insufficient data.
"""

from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _infer_master_airports(
    df_flights: pd.DataFrame, df_airports: Optional[pd.DataFrame]
) -> List[str]:
    if df_airports is not None and "iata_code" in df_airports.columns:
        master = sorted(df_airports["iata_code"].dropna().unique().tolist())
    else:
        deps = (
            df_flights["dep_airport"].dropna().unique().tolist()
            if "dep_airport" in df_flights.columns
            else []
        )
        arrs = (
            df_flights["arr_airport"].dropna().unique().tolist()
            if "arr_airport" in df_flights.columns
            else []
        )
        master = sorted(set(deps) | set(arrs))
    return master


def _floor_times(series: pd.Series, freq: str) -> pd.Series:
    # freq: 'H' or 'D'
    if freq == "H":
        return pd.to_datetime(series, errors="coerce").dt.floor("h")
    return pd.to_datetime(series, errors="coerce").dt.floor("D")


def _counts_by_time(
    df: pd.DataFrame,
    time_col: str,
    airport_col: str,
    freq: str,
    time_index: pd.DatetimeIndex,
    airports: List[str],
) -> pd.DataFrame:
    s = df.copy()
    if time_col not in s.columns or airport_col not in s.columns:
        # return zero-filled frame
        return pd.DataFrame(0, index=time_index, columns=airports).astype(int)

    s[time_col] = pd.to_datetime(s[time_col], errors="coerce")
    s = s.dropna(subset=[time_col, airport_col])
    if s.empty:
        return pd.DataFrame(0, index=time_index, columns=airports).astype(int)

    time_floor = "h" if freq == "H" else "D"
    s["time_bucket"] = s[time_col].dt.floor(time_floor)
    grouped = s.groupby(["time_bucket", airport_col]).size().unstack(fill_value=0)

    # reindex to full time index and airport columns
    grouped = grouped.reindex(index=time_index, columns=airports, fill_value=0)
    return grouped.astype(int)


def compute_concurrency_with_coverage(
    df_flights: pd.DataFrame,
    df_airports: Optional[pd.DataFrame] = None,
    freq: str = "H",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    include_master_only: bool = True,
    min_hours_covered: int = 0,
    by_groups: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute concurrency (arrivals/departures) plus per-airport coverage.

    Returns (counts_time, counts_day, coverage_df).

    Parameters
    - df_flights: flights with 'std' and 'sta' and airport columns (dep_airport/arr_airport or similar)
    - df_airports: optional master list with 'iata_code' column
    - freq: 'H' or 'D' for hourly/daily counts
    - start/end: optional datetimes to bound the time series
    - include_master_only: if True, the output airports set will be the master airport list; otherwise
      include any airport observed in the flights data as well.
    - min_hours_covered: post-filter threshold: airports with fewer than this many non-empty hour-buckets
      will be flagged in coverage but not automatically removed (you can use coverage_df to filter downstream)
    - by_groups: if True, aggregate by airport groups instead of individual airports
    """
    assert freq in ("H", "D"), "freq must be 'H' or 'D'"

    # Normalize some common column names
    flights = df_flights.copy()
    # Common airport column alternatives
    dep_col = next(
        (
            c
            for c in ["dep_airport", "origin", "departure_airport", "dep"]
            if c in flights.columns
        ),
        "dep_airport",
    )
    arr_col = next(
        (
            c
            for c in ["arr_airport", "destination", "arrival_airport", "arr"]
            if c in flights.columns
        ),
        "arr_airport",
    )

    # Handle airport groups aggregation
    if by_groups:
        dep_group_col = next(
            (c for c in ["dep_airport_group", "origin_group"] if c in flights.columns),
            "dep_airport_group",
        )
        arr_group_col = next(
            (
                c
                for c in ["arr_airport_group", "destination_group"]
                if c in flights.columns
            ),
            "arr_airport_group",
        )

        # Use groups as the aggregation unit
        dep_col = dep_group_col
        arr_col = arr_group_col

        # Master list is unique groups
        dep_groups = (
            flights[dep_group_col].dropna().unique().tolist()
            if dep_group_col in flights.columns
            else []
        )
        arr_groups = (
            flights[arr_group_col].dropna().unique().tolist()
            if arr_group_col in flights.columns
            else []
        )
        master = sorted(set(dep_groups) | set(arr_groups))
    else:
        # Master airports list
        master = _infer_master_airports(flights, df_airports)

    # If include_master_only is False, extend master with any observed airports in flights
    if not include_master_only:
        observed = set()
        if dep_col in flights.columns:
            observed.update(flights[dep_col].dropna().unique().tolist())
        if arr_col in flights.columns:
            observed.update(flights[arr_col].dropna().unique().tolist())
        master = sorted(set(master) | observed)

    # Determine time bounds
    stds = pd.to_datetime(flights.get("std", pd.Series([])), errors="coerce")
    stas = pd.to_datetime(flights.get("sta", pd.Series([])), errors="coerce")
    min_time = min(
        (t for t in [stds.min(skipna=True), stas.min(skipna=True)] if pd.notna(t)),
        default=None,
    )
    max_time = max(
        (t for t in [stds.max(skipna=True), stas.max(skipna=True)] if pd.notna(t)),
        default=None,
    )

    if min_time is None or max_time is None:
        raise ValueError("No valid scheduled times (std/sta) found in flights")

    start_ts = (
        pd.to_datetime(start)
        if start is not None
        else pd.to_datetime(min_time).floor("D")
    )
    end_ts = (
        pd.to_datetime(end) if end is not None else pd.to_datetime(max_time).ceil("D")
    )

    # Convert freq for pandas compatibility ('H' -> 'h')
    pandas_freq = "h" if freq == "H" else freq
    time_index = pd.date_range(start=start_ts, end=end_ts, freq=pandas_freq)
    daily_index = pd.date_range(
        start=start_ts.floor("D"), end=end_ts.floor("D"), freq="D"
    )

    # Counts
    deps = _counts_by_time(flights, "std", dep_col, freq, time_index, master)
    arrs = _counts_by_time(flights, "sta", arr_col, freq, time_index, master)

    # Label columns
    deps.columns = pd.MultiIndex.from_product([deps.columns.tolist(), ["departures"]])
    arrs.columns = pd.MultiIndex.from_product([arrs.columns.tolist(), ["arrivals"]])

    counts_time = pd.concat([deps, arrs], axis=1).sort_index(axis=1)

    # Daily aggregates
    deps_daily = _counts_by_time(flights, "std", dep_col, "D", daily_index, master)
    arrs_daily = _counts_by_time(flights, "sta", arr_col, "D", daily_index, master)
    deps_daily.columns = pd.MultiIndex.from_product(
        [deps_daily.columns.tolist(), ["departures"]]
    )
    arrs_daily.columns = pd.MultiIndex.from_product(
        [arrs_daily.columns.tolist(), ["arrivals"]]
    )
    counts_day = pd.concat([deps_daily, arrs_daily], axis=1).sort_index(axis=1)

    # Coverage summary
    coverage = []
    total_buckets = len(time_index)
    entity_type = "group" if by_groups else "airport"
    for entity in master:
        # count non-zero hourly buckets where entity has any event
        if entity in counts_time.columns.get_level_values(0):
            # select both departures and arrivals for this entity
            try:
                dep_series = counts_time[(entity, "departures")]
            except KeyError:
                dep_series = pd.Series(0, index=time_index)
            try:
                arr_series = counts_time[(entity, "arrivals")]
            except KeyError:
                arr_series = pd.Series(0, index=time_index)

            any_event = (dep_series.fillna(0) + arr_series.fillna(0)) > 0
            hours_covered = int(any_event.sum())
            total_events = int(dep_series.sum()) + int(arr_series.sum())
            coverage.append(
                {
                    entity_type: entity,
                    "hours_covered": hours_covered,
                    "total_hours": total_buckets,
                    "hours_covered_ratio": (
                        hours_covered / total_buckets if total_buckets else 0.0
                    ),
                    "total_events": total_events,
                }
            )
        else:
            coverage.append(
                {
                    entity_type: entity,
                    "hours_covered": 0,
                    "total_hours": total_buckets,
                    "hours_covered_ratio": 0.0,
                    "total_events": 0,
                }
            )

    coverage_df = pd.DataFrame(coverage).set_index(entity_type)
    coverage_df["sufficient_coverage"] = (
        coverage_df["hours_covered"] >= min_hours_covered
    )

    return counts_time, counts_day, coverage_df


def create_concurrency_lookup_dataset(
    df_flights: pd.DataFrame,
    df_airports: Optional[pd.DataFrame] = None,
    data_dir: Optional[Path] = None,
    output_filename: str = "concurrency_lookup.csv",
    freq: str = "H",
    by_groups: bool = True,
    include_both: bool = True,
) -> Path:
    """Create and save a concurrency lookup dataset for fast feature engineering.

    This creates a flattened lookup table with both airport and group level data.
    If include_both=True, generates entries for both individual airports and groups.

    Returns the path to the saved dataset.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    output_path = data_dir / output_filename

    print(f"Computing concurrency lookup dataset...")

    # Load airport group mapping
    airport_group_map = {}
    if data_dir:
        airport_groups_file = data_dir / "avinor_data/airportgroups.csv"
        if airport_groups_file.exists():
            try:
                df_groups = pd.read_csv(airport_groups_file)
                if (
                    "airport" in df_groups.columns
                    and "airport_group" in df_groups.columns
                ):
                    airport_group_map = df_groups.set_index("airport")[
                        "airport_group"
                    ].to_dict()
                    print(f"   Loaded {len(airport_group_map)} airport group mappings")
            except Exception as e:
                print(f"   Warning: Could not load airport groups: {e}")

    all_lookup_records = []

    # Generate airport-level data if requested
    if include_both or not by_groups:
        print("   Computing airport-level concurrency...")
        counts_time_airports, _, _ = compute_concurrency_with_coverage(
            df_flights=df_flights,
            df_airports=df_airports,
            freq=freq,
            by_groups=False,  # Individual airports
            include_master_only=True,
        )

        # Process airport-level data
        for timestamp in counts_time_airports.index:
            timestamp_data = counts_time_airports.loc[timestamp]
            entities = timestamp_data.index.get_level_values(0).unique()

            for entity in entities:
                try:
                    dep_count = timestamp_data.get((entity, "departures"), 0)
                    arr_count = timestamp_data.get((entity, "arrivals"), 0)

                    record = {
                        "datetime": timestamp,
                        "entity": entity,
                        "dep_concurrency": int(dep_count) if pd.notna(dep_count) else 0,
                        "arr_concurrency": int(arr_count) if pd.notna(arr_count) else 0,
                        "airport": entity,  # Individual airport code
                        "airport_group": airport_group_map.get(entity, entity),  # Map to group
                    }

                    all_lookup_records.append(record)

                except Exception as e:
                    print(f"Warning: Error processing airport {entity} at {timestamp}: {e}")
                    continue

    # Generate group-level data if requested
    if include_both or by_groups:
        print("   Computing group-level concurrency...")
        counts_time_groups, _, _ = compute_concurrency_with_coverage(
            df_flights=df_flights,
            df_airports=df_airports,
            freq=freq,
            by_groups=True,  # Airport groups
            include_master_only=True,
        )

        # Process group-level data
        for timestamp in counts_time_groups.index:
            timestamp_data = counts_time_groups.loc[timestamp]
            entities = timestamp_data.index.get_level_values(0).unique()

            for entity in entities:
                try:
                    dep_count = timestamp_data.get((entity, "departures"), 0)
                    arr_count = timestamp_data.get((entity, "arrivals"), 0)

                    record = {
                        "datetime": timestamp,
                        "entity": entity,
                        "dep_concurrency": int(dep_count) if pd.notna(dep_count) else 0,
                        "arr_concurrency": int(arr_count) if pd.notna(arr_count) else 0,
                        "airport": None,  # No individual airport for groups
                        "airport_group": entity,  # Group identifier
                    }

                    all_lookup_records.append(record)

                except Exception as e:
                    print(f"Warning: Error processing group {entity} at {timestamp}: {e}")
                    continue

    # Create DataFrame
    lookup_df = pd.DataFrame(all_lookup_records)

    if lookup_df.empty:
        print("Warning: No concurrency data generated")
        return output_path

    # Sort and clean
    lookup_df = lookup_df.sort_values(["datetime", "entity"]).reset_index(drop=True)

    # Save dataset
    lookup_df.to_csv(output_path, index=False)

    print(f"Concurrency lookup dataset saved: {output_path}")
    print(f"   Shape: {lookup_df.shape}")
    print(
        f"   Time range: {lookup_df['datetime'].min()} to {lookup_df['datetime'].max()}"
    )
    print(f"   Entities: {lookup_df['entity'].nunique()}")
    airport_entries = lookup_df[lookup_df['airport'].notna()]
    group_entries = lookup_df[lookup_df['airport'].isna()]
    print(f"   Airport-level entries: {len(airport_entries)} ({airport_entries['airport'].nunique()} unique airports)")
    print(f"   Group-level entries: {len(group_entries)} ({group_entries['airport_group'].nunique()} unique groups)")

    return output_path


def load_concurrency_lookup(
    data_dir: Optional[Path] = None, filename: str = "concurrency_lookup.csv"
) -> Optional[pd.DataFrame]:
    """Load pre-computed concurrency lookup dataset.

    The dataset contains both airport-level and group-level concurrency data:
    - Rows with airport != None: individual airport concurrency
    - Rows with airport == None: airport group concurrency

    Returns None if file doesn't exist.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    lookup_path = data_dir / filename

    if not lookup_path.exists():
        return None

    try:
        df = pd.read_csv(lookup_path)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        print(f"✅ Loaded concurrency lookup: {lookup_path} (shape: {df.shape})")
        return df
    except Exception as e:
        print(f"⚠️ Error loading concurrency lookup {lookup_path}: {e}")
        return None


def save_concurrency_tables(
    counts_time: pd.DataFrame,
    counts_day: pd.DataFrame,
    out_dir: Optional[Path] = None,
    prefix: str = "concurrency",
) -> Tuple[Path, Path]:
    out_dir = Path(out_dir) if out_dir is not None else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    def _flatten(df: pd.DataFrame) -> pd.DataFrame:
        flat = df.copy()
        flat.columns = [f"{a}_{t}" for a, t in flat.columns.to_list()]
        return flat

    time_file = out_dir / f"{prefix}_hourly.csv"
    day_file = out_dir / f"{prefix}_daily.csv"
    _flatten(counts_time).to_csv(time_file, index=True)
    _flatten(counts_day).to_csv(day_file, index=True)
    return time_file, day_file


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Compute airport concurrency and coverage.")
    p.add_argument("--freq", choices=["H", "D"], default="H")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--save", action="store_true")
    p.add_argument(
        "--create-dataset",
        action="store_true",
        help="Create concurrency lookup dataset for fast feature engineering",
    )
    p.add_argument("--include-master-only", action="store_true")
    p.add_argument("--min-hours-covered", type=int, default=0)
    p.add_argument(
        "--by-groups",
        action="store_true",
        help="Aggregate by airport groups instead of individual airports",
    )
    args = p.parse_args()

    flights_file = DATA_DIR / "avinor_data/historical_flights.csv"
    airports_file = DATA_DIR / "airport_coordinates.csv"

    if not flights_file.exists():
        raise SystemExit(f"No flights file: {flights_file}")

    df_f = pd.read_csv(flights_file)
    df_ap = pd.read_csv(airports_file) if airports_file.exists() else None

    if args.create_dataset:
        # Create concurrency lookup dataset
        dataset_path = create_concurrency_lookup_dataset(
            df_f, df_ap, freq=args.freq, by_groups=args.by_groups, data_dir=DATA_DIR
        )
        print(f"✅ Created concurrency lookup dataset: {dataset_path}")
    else:
        # Original functionality
        counts_time, counts_day, coverage_df = compute_concurrency_with_coverage(
            df_f,
            df_ap,
            freq=args.freq,
            start=args.start,
            end=args.end,
            include_master_only=args.include_master_only,
            min_hours_covered=args.min_hours_covered,
            by_groups=args.by_groups,
        )

        print("Coverage summary (top rows):")
        print(coverage_df.head())

        if args.save:
            tfile, dfile = save_concurrency_tables(counts_time, counts_day)
            print(f"Saved: {tfile}, {dfile}")
        else:
            print("Computed concurrency tables (not saved).")
