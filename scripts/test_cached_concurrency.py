#!/usr/bin/env python3
"""
Test script to demonstrate the fast concurrency feature system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_engineering import build_flight_weather_features
from src.concurrency import load_concurrency_lookup
import pandas as pd
import time

def main():
    """Test the cached concurrency system"""
    data_dir = Path(__file__).parent.parent / "data"
    
    print("🧪 Testing cached concurrency feature system...")
    
    # Check if concurrency dataset exists
    concurrency_df = load_concurrency_lookup(data_dir)
    
    if concurrency_df is not None:
        print("✅ Found cached concurrency data!")
        print(f"   Shape: {concurrency_df.shape}")
        print(f"   Time range: {concurrency_df['datetime'].min()} to {concurrency_df['datetime'].max()}")
        print(f"   Entities: {concurrency_df['entity'].nunique()}")
        
        # Test with small sample
        print("\n🚀 Testing feature engineering with cached concurrency...")
        start_time = time.time()
        
        # Load small sample of flight data
        flights_file = data_dir / "historical_flights.csv"
        if flights_file.exists():
            df_flights = pd.read_csv(flights_file).head(1000)  # Small sample for testing
            print(f"   Using {len(df_flights)} flight records for test")
            
            result = build_flight_weather_features(
                df_flights=df_flights,
                include_concurrency_features=True
            )
            
            end_time = time.time()
            
            print(f"✅ Success! Processing took {end_time - start_time:.2f} seconds")
            print(f"   Final shape: {result.shape}")
            
            # Check concurrency columns
            concurrency_cols = [col for col in result.columns if 'departures_hour' in col or 'arrivals_hour' in col]
            if concurrency_cols:
                print(f"   Concurrency features added: {len(concurrency_cols)}")
                print(f"   Sample concurrency columns: {concurrency_cols[:4]}")
                
                # Show some stats
                for col in concurrency_cols[:2]:
                    non_zero = (result[col] > 0).sum()
                    print(f"   {col}: {non_zero:,} non-zero values (max: {result[col].max()})")
            else:
                print("   ⚠️ No concurrency features found")
        else:
            print(f"   ⚠️ Flight data not found at {flights_file}")
            
    else:
        print("⚠️ No cached concurrency data found")
        print("   Run: python scripts/create_concurrency_dataset.py")
        print("   Or: python -m src.concurrency --create-dataset --by-groups")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())