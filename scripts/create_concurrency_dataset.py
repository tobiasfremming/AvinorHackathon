#!/usr/bin/env python3
"""
Script to pre-compute concurrency lookup dataset for fast feature engineering.

Usage:
    python scripts/create_concurrency_dataset.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.concurrency import create_concurrency_lookup_dataset
import pandas as pd


def main():
    """Create concurrency lookup dataset"""
    # Define data directory
    data_dir = Path(__file__).parent.parent / "data"

    # Load flight and airport data
    flights_file = data_dir / "avinor_data/historical_flights.csv"
    airports_file = data_dir / "airport_coordinates.csv"

    if not flights_file.exists():
        print(f"Error: Flight data not found at {flights_file}")
        return 1

    print(f"Loading flight data from {flights_file}")
    df_flights = pd.read_csv(flights_file)

    df_airports = None
    if airports_file.exists():
        print(f"✈️ Loading airport data from {airports_file}")
        df_airports = pd.read_csv(airports_file)
    else:
        print(f"Airport data not found at {airports_file}, will infer from flights")

    # Create concurrency dataset
    print("Creating concurrency lookup dataset...")

    try:
        dataset_path = create_concurrency_lookup_dataset(
            df_flights=df_flights,
            df_airports=df_airports,
            data_dir=data_dir,
            freq="H",
            by_groups=True,
        )

        print(f"Success! Concurrency dataset created at: {dataset_path}")

        # Also create airport-level dataset
        airport_dataset_path = create_concurrency_lookup_dataset(
            df_flights=df_flights,
            df_airports=df_airports,
            data_dir=data_dir,
            output_filename="concurrency_lookup_airports.csv",
            freq="H",
            by_groups=False,
        )

        print(f"Airport-level dataset created at: {airport_dataset_path}")

        return 0

    except Exception as e:
        print(f"Error creating concurrency dataset: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
