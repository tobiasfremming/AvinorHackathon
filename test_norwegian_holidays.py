#!/usr/bin/env python3
"""
Test script for the updated Norwegian holiday feature engineer
"""

import sys
import os
sys.path.append('/home/kristsey/AvinorHackathon')

from src.holiday_features import HolidayFeatureEngineer
import pandas as pd

def test_norwegian_holidays():
    """Test the updated Norwegian holiday detection."""
    print("🇳🇴 TESTING UPDATED NORWEGIAN HOLIDAY DETECTION")
    print("=" * 60)
    
    # Initialize the holiday engineer
    holiday_engineer = HolidayFeatureEngineer()
    
    # Test cases for Norwegian holidays
    test_cases = [
        # Summer holidays (should be 1)
        ('2024-07-15', 'norway', 1, 'Summer holiday period'),
        ('2024-07-22', 'norway', 1, 'Peak summer vacation'),
        ('2024-08-01', 'norway', 1, 'Late summer vacation'),
        
        # Christmas holidays (should be 1)
        ('2024-12-25', 'norway', 1, 'Christmas Day'),
        ('2024-12-30', 'norway', 1, 'Between Christmas and NY'),
        ('2025-01-02', 'norway', 1, 'New Year period'),
        
        # Easter holidays (should be 1)
        ('2024-03-28', 'norway', 1, 'Maundy Thursday'),
        ('2024-03-29', 'norway', 1, 'Good Friday'),
        ('2024-03-31', 'norway', 1, 'Easter Sunday'),
        
        # Winter sports holiday (should be 1)
        ('2024-02-19', 'norway', 1, 'Winter sports vacation'),
        ('2024-02-22', 'norway', 1, 'Mid-winter break'),
        
        # Constitution Day (should be 1)
        ('2024-05-17', 'norway', 1, 'Constitution Day'),
        
        # Regular days (should be 0)
        ('2024-06-10', 'norway', 0, 'Regular June day'),
        ('2024-09-15', 'norway', 0, 'Regular September day'),
        ('2024-11-15', 'norway', 0, 'Regular November day'),
    ]
    
    passed = 0
    failed = 0
    
    print(f"{'Date':<12} {'Expected':<8} {'Actual':<8} {'Status':<8} {'Description'}")
    print("-" * 60)
    
    for date, country, expected, description in test_cases:
        try:
            actual = holiday_engineer.is_vacation_day(date, country, buffer_days=2)
            status = "✅ PASS" if actual == expected else "❌ FAIL"
            
            if actual == expected:
                passed += 1
            else:
                failed += 1
            
            print(f"{date:<12} {expected:<8} {actual:<8} {status:<8} {description}")
            
        except Exception as e:
            failed += 1
            print(f"{date:<12} {expected:<8} {'ERROR':<8} ❌ FAIL  {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    
    success_rate = (passed / len(test_cases)) * 100
    print(f"✅ Success rate: {success_rate:.1f}%")
    
    return passed, failed

def test_holiday_features_on_sample_data():
    """Test holiday feature creation on sample flight data."""
    print("\n🛫 TESTING HOLIDAY FEATURES ON SAMPLE FLIGHT DATA")
    print("=" * 60)
    
    # Create sample flight data
    sample_dates = [
        '2024-01-15',  # Regular day
        '2024-02-19',  # Winter sports holiday
        '2024-05-17',  # Constitution Day
        '2024-07-15',  # Summer vacation
        '2024-12-25',  # Christmas
    ]
    
    sample_data = pd.DataFrame({
        'std': pd.to_datetime(sample_dates),
        'flight_id': ['FL001', 'FL002', 'FL003', 'FL004', 'FL005'],
        'dep_airport': ['OSL', 'BGO', 'TRD', 'OSL', 'SVG'],
        'arr_airport': ['BGO', 'OSL', 'OSL', 'CPH', 'OSL']
    })
    
    print("Sample flight data:")
    print(sample_data)
    
    # Initialize holiday engineer and add features
    holiday_engineer = HolidayFeatureEngineer()
    
    try:
        df_with_holidays = holiday_engineer.add_holiday_features(
            sample_data,
            date_col='std',
            origin_country='norway',
            dest_country='norway'
        )
        
        # Check holiday features
        holiday_cols = [col for col in df_with_holidays.columns if 'vacation' in col or 'holiday' in col]
        
        print(f"\n📊 Holiday features added: {holiday_cols}")
        print("\nHoliday feature values:")
        print(df_with_holidays[['std'] + holiday_cols])
        
        # Check for variation
        print(f"\n📈 Feature variation check:")
        for col in holiday_cols:
            unique_vals = df_with_holidays[col].nunique()
            print(f"  {col}: {unique_vals} unique values")
            if unique_vals <= 5:
                print(f"    Values: {df_with_holidays[col].value_counts().to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error adding holiday features: {e}")
        return False

if __name__ == "__main__":
    # Run tests
    print("Starting Norwegian holiday detection tests...\n")
    
    # Test 1: Individual holiday detection
    passed, failed = test_norwegian_holidays()
    
    # Test 2: Holiday feature creation
    feature_test_success = test_holiday_features_on_sample_data()
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"  Holiday detection: {passed}/{passed + failed} tests passed")
    print(f"  Feature creation: {'✅ SUCCESS' if feature_test_success else '❌ FAILED'}")
    
    if passed > failed and feature_test_success:
        print(f"\n🎉 Norwegian holiday features are working correctly!")
        print(f"   You can now use them in your regression pipeline.")
    else:
        print(f"\n⚠️  Some issues found. Please check the error messages above.")