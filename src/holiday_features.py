"""
Holiday Feature Engineering for Flight Delay Prediction
Integrates vacation/holiday data into flight prediction models
"""

import datetime
import pandas as pd
import requests
import json
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HolidayFeatureEngineer:
    """
    A class to create holiday-related features for flight delay prediction.
    Optimized for Norwegian holidays using vectorized pandas operations.
    """
    
    def __init__(self):
        self.supported_countries = self._get_supported_countries()
        self._holiday_cache = {}
        # Load and cache all Norwegian holiday data once
        self._load_all_norwegian_data()
    
    def _load_all_norwegian_data(self):
        """Load and cache all Norwegian holiday data once for efficient vectorized operations."""
        try:
            # Load public holidays
            csv_path = "/home/kristsey/AvinorHackathon/data/helligdager_stavanger.csv"
            self.norwegian_holidays_df = pd.read_csv(csv_path)
            self.norwegian_holidays_df['date'] = pd.to_datetime(self.norwegian_holidays_df['date'])
            
            # Create school vacation periods dataframe
            self.norwegian_vacations_df = self._create_school_vacations_df()
            
            # Combine all Norwegian holiday/vacation dates with buffer
            self._create_norwegian_holiday_lookup()
            
            logger.info(f"Cached {len(self.norwegian_holidays_df)} public holidays and {len(self.norwegian_vacations_df)} vacation periods")
            
        except Exception as e:
            logger.error(f"Error loading Norwegian holiday data: {e}")
            self.norwegian_holidays_df = pd.DataFrame()
            self.norwegian_vacations_df = pd.DataFrame()
    
    def _create_school_vacations_df(self) -> pd.DataFrame:
        """Create a DataFrame of all Norwegian school vacation periods."""
        vacation_data = []
        
        for year in range(2018, 2028):  # Cover full range
            # Summer vacation
            vacation_data.append({
                'year': year,
                'vacation_type': 'summer',
                'start_date': f'{year}-06-20',
                'end_date': f'{year}-08-15',
                'name': 'Summer vacation (Sommerferie)'
            })
            
            # Christmas vacation
            vacation_data.append({
                'year': year,
                'vacation_type': 'christmas',
                'start_date': f'{year}-12-20',
                'end_date': f'{year+1}-01-08',
                'name': 'Christmas vacation (Juleferie)'
            })
            
            # Winter sports vacation
            winter_dates = self._get_winter_sports_dates(year)
            vacation_data.append({
                'year': year,
                'vacation_type': 'winter_sports',
                'start_date': winter_dates['start'],
                'end_date': winter_dates['end'],
                'name': 'Winter sports vacation (Vinterferie)'
            })
            
            # Autumn break
            autumn_dates = self._get_autumn_break_dates(year)
            vacation_data.append({
                'year': year,
                'vacation_type': 'autumn',
                'start_date': autumn_dates['start'],
                'end_date': autumn_dates['end'],
                'name': 'Autumn break (Høstferie)'
            })
        
        df = pd.DataFrame(vacation_data)
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        
        return df
    
    def _create_norwegian_holiday_lookup(self):
        """Create a comprehensive lookup set of all Norwegian holiday dates with buffer."""
        holiday_dates = set()
        
        # Add public holidays with buffer
        for _, row in self.norwegian_holidays_df.iterrows():
            base_date = row['date']
            for buffer in range(-2, 3):  # ±2 day buffer
                holiday_dates.add((base_date + pd.Timedelta(days=buffer)).date())
        
        # Add vacation periods with buffer
        for _, row in self.norwegian_vacations_df.iterrows():
            start_date = (row['start_date'] - pd.Timedelta(days=2)).date()  # Buffer
            end_date = (row['end_date'] + pd.Timedelta(days=2)).date()      # Buffer
            
            current_date = start_date
            while current_date <= end_date:
                holiday_dates.add(current_date)
                current_date += datetime.timedelta(days=1)
        
        self.norwegian_holiday_dates = holiday_dates
        logger.info(f"Created holiday lookup with {len(holiday_dates)} dates")
        
    def _get_supported_countries(self):
        """Get all countries supported - includes Norway via CSV data."""
        return {
            'albania': 'AL', 'andorra': 'AD', 'austria': 'AT', 'belarus': 'BY', 'belgium': 'BE',
            'brazil': 'BR', 'bulgaria': 'BG', 'croatia': 'HR', 'czechia': 'CZ', 'estonia': 'EE',
            'france': 'FR', 'germany': 'DE', 'hungary': 'HU', 'ireland': 'IE', 'italy': 'IT',
            'latvia': 'LV', 'liechtenstein': 'LI', 'lithuania': 'LT', 'luxembourg': 'LU',
            'malta': 'MT', 'mexico': 'MX', 'moldova': 'MD', 'monaco': 'MC', 'netherlands': 'NL',
            'norway': 'NO',  # Supported via CSV data
            'poland': 'PL', 'portugal': 'PT', 'romania': 'RO', 'san marino': 'SM', 'serbia': 'RS',
            'slovakia': 'SK', 'slovenia': 'SI', 'south africa': 'ZA', 'spain': 'ES',
            'sweden': 'SE', 'switzerland': 'CH', 'vatican city': 'VA',
            # Common aliases
            'czech republic': 'CZ', 'vatican': 'VA'
        }
    
    @lru_cache(maxsize=128)
    def _fetch_school_holidays_cached(self, country_code: str, year: int) -> tuple:
        """Cached version of holiday fetching to avoid repeated API calls."""
        holidays = self._fetch_school_holidays(country_code, year)
        return tuple((h.get('startDate', ''), h.get('endDate', ''), h.get('name', [{}])[0].get('text', '')) 
                     for h in holidays if isinstance(h, dict))
    
    def _load_norwegian_holidays_from_csv(self) -> pd.DataFrame:
        """Load Norwegian public holidays from the actual helligdager_stavanger.csv file."""
        try:
            csv_path = "/home/kristsey/AvinorHackathon/data/helligdager_stavanger.csv"
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"Loaded {len(df)} Norwegian holidays from helligdager_stavanger.csv")
            return df
        except Exception as e:
            logger.error(f"Could not load helligdager_stavanger.csv: {e}")
            return None
    
    def _get_norwegian_school_vacations(self, year: int) -> List[Dict]:
        """Get Norwegian school vacation periods manually since they're not in public holiday data."""
        vacations = []
        
        try:
            # Summer vacation (Sommerferie) - Week 26-33 approximately
            vacations.append({
                'startDate': f'{year}-06-20',
                'endDate': f'{year}-08-15', 
                'name': [{'text': 'Summer vacation (Sommerferie)'}]
            })
            
            # Christmas vacation (Juleferie) - Dec 20 to Jan 8
            vacations.append({
                'startDate': f'{year}-12-20',
                'endDate': f'{year+1}-01-08',
                'name': [{'text': 'Christmas vacation (Juleferie)'}]
            })
            
            # Winter sports vacation (Vinterferie) - Week 8 (mid-February)
            winter_dates = self._get_winter_sports_dates(year)
            vacations.append({
                'startDate': winter_dates['start'],
                'endDate': winter_dates['end'],
                'name': [{'text': 'Winter sports vacation (Vinterferie)'}]
            })
            
            # Autumn break (Høstferie) - Week 40 (early October)
            autumn_dates = self._get_autumn_break_dates(year)
            vacations.append({
                'startDate': autumn_dates['start'],
                'endDate': autumn_dates['end'],
                'name': [{'text': 'Autumn break (Høstferie)'}]
            })
            
            logger.info(f"Added {len(vacations)} school vacation periods for {year}")
            
        except Exception as e:
            logger.error(f"Error creating school vacations for {year}: {e}")
        
        return vacations
    
    def _get_winter_sports_dates(self, year: int) -> Dict[str, str]:
        """Get winter sports holiday dates (Week 8, mid-February)."""
        winter_dates = {
            2018: {'start': '2018-02-19', 'end': '2018-02-23'},
            2019: {'start': '2019-02-18', 'end': '2019-02-22'},
            2020: {'start': '2020-02-17', 'end': '2020-02-21'},
            2021: {'start': '2021-02-15', 'end': '2021-02-19'},
            2022: {'start': '2022-02-21', 'end': '2022-02-25'},
            2023: {'start': '2023-02-20', 'end': '2023-02-24'},
            2024: {'start': '2024-02-19', 'end': '2024-02-23'},
            2025: {'start': '2025-02-17', 'end': '2025-02-21'},
            2026: {'start': '2026-02-16', 'end': '2026-02-20'},
            2027: {'start': '2027-02-15', 'end': '2027-02-19'},
        }
        return winter_dates.get(year, {'start': f'{year}-02-19', 'end': f'{year}-02-23'})
    
    def _get_autumn_break_dates(self, year: int) -> Dict[str, str]:
        """Get autumn break dates (Week 40, early October)."""
        autumn_dates = {
            2018: {'start': '2018-10-01', 'end': '2018-10-05'},
            2019: {'start': '2019-09-30', 'end': '2019-10-04'},
            2020: {'start': '2020-10-05', 'end': '2020-10-09'},
            2021: {'start': '2021-10-04', 'end': '2021-10-08'},
            2022: {'start': '2022-10-03', 'end': '2022-10-07'},
            2023: {'start': '2023-10-02', 'end': '2023-10-06'},
            2024: {'start': '2024-09-30', 'end': '2024-10-04'},
            2025: {'start': '2025-10-06', 'end': '2025-10-10'},
            2026: {'start': '2026-10-05', 'end': '2026-10-09'},
            2027: {'start': '2027-10-04', 'end': '2027-10-08'},
        }
        return autumn_dates.get(year, {'start': f'{year}-10-01', 'end': f'{year}-10-05'})
    
    def _fetch_school_holidays(self, country_code: str, year: int) -> List[Dict]:
        """Fetch school holidays - use CSV data for Norway, API for others."""
        if country_code.lower() in ['norway', 'no']:
            return self._get_norwegian_holidays_from_csv(year)
        else:
            # Use OpenHolidaysAPI for other supported countries
            if country_code.lower() in self.supported_countries:
                iso_code = self.supported_countries[country_code.lower()]
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
                    logger.warning(f"API request failed for {country_code} {year}: {response.status_code}")
                    return []
            except Exception as e:
                logger.error(f"Error fetching holidays for {country_code} {year}: {e}")
                return []
    
    def _get_norwegian_holidays_from_csv(self, year: int) -> List[Dict]:
        """Get Norwegian holidays from helligdager_stavanger.csv for a specific year."""
        holidays = []
        
        try:
            # Load public holidays from the actual CSV
            df_holidays = self._load_norwegian_holidays_from_csv()
            if df_holidays is not None:
                # Filter by year
                year_holidays = df_holidays[df_holidays['year'] == year]
                
                # Convert to expected format
                for _, row in year_holidays.iterrows():
                    holidays.append({
                        'startDate': row['date'].strftime('%Y-%m-%d'),
                        'endDate': row['date'].strftime('%Y-%m-%d'),
                        'name': [{'text': row['holiday_name']}]
                    })
                
                logger.info(f"Loaded {len(year_holidays)} public holidays for {year} from helligdager_stavanger.csv")
            
            # Add school vacation periods manually
            school_vacations = self._get_norwegian_school_vacations(year)
            holidays.extend(school_vacations)
            
            logger.info(f"Total {len(holidays)} holidays/vacations for {year}")
            return holidays
            
        except Exception as e:
            logger.error(f"Error loading Norwegian holidays from CSV for {year}: {e}")
            # Fallback to manual calendar if CSV fails
            return self._get_norwegian_holidays_manual_fallback(year)
    
    def _get_norwegian_holidays_manual_fallback(self, year: int) -> List[Dict]:
        """Fallback manual Norwegian holiday calendar if CSV loading fails."""
        logger.warning(f"Using fallback manual Norwegian calendar for {year}")
        
        holidays = [
            # Summer vacation
            {'startDate': f'{year}-06-20', 'endDate': f'{year}-08-15', 'name': [{'text': 'Summer vacation'}]},
            # Christmas vacation  
            {'startDate': f'{year}-12-20', 'endDate': f'{year+1}-01-08', 'name': [{'text': 'Christmas vacation'}]},
            # Winter sports (approximate)
            {'startDate': f'{year}-02-17', 'endDate': f'{year}-02-21', 'name': [{'text': 'Winter sports vacation'}]},
            # Autumn break
            {'startDate': f'{year}-10-01', 'endDate': f'{year}-10-05', 'name': [{'text': 'Autumn break'}]},
            # Basic public holidays
            {'startDate': f'{year}-01-01', 'endDate': f'{year}-01-01', 'name': [{'text': 'New Year\'s Day'}]},
            {'startDate': f'{year}-05-01', 'endDate': f'{year}-05-01', 'name': [{'text': 'Labour Day'}]},
            {'startDate': f'{year}-05-17', 'endDate': f'{year}-05-17', 'name': [{'text': 'Constitution Day'}]},
            {'startDate': f'{year}-12-25', 'endDate': f'{year}-12-25', 'name': [{'text': 'Christmas Day'}]},
            {'startDate': f'{year}-12-26', 'endDate': f'{year}-12-26', 'name': [{'text': 'Boxing Day'}]},
        ]
        
        return holidays
    
    def is_vacation_day(self, date, country_code: str, buffer_days: int = 2) -> int:
        """
        Check if a given date falls within a vacation period.
        Optimized for Norwegian holidays using cached lookup.
        """
        if country_code.lower() not in ['norway', 'no']:
            logger.warning(f"Only Norway is supported for optimized lookup, got {country_code}")
            return 0
        
        try:
            if isinstance(date, str):
                date_obj = datetime.datetime.strptime(date, '%Y-%m-%d').date()
            elif hasattr(date, 'date'):
                date_obj = date.date()
            else:
                date_obj = pd.to_datetime(date).date()
            
            # Fast lookup using cached set
            return 1 if date_obj in self.norwegian_holiday_dates else 0
            
        except Exception as e:
            logger.error(f"Error in is_vacation_day for {date}: {e}")
            return 0
    
    def is_vacation_day_vectorized(self, dates: pd.Series, country_code: str = 'norway') -> pd.Series:
        """
        Vectorized version of is_vacation_day for pandas Series.
        Much faster for large datasets.
        """
        if country_code.lower() not in ['norway', 'no']:
            logger.warning(f"Only Norway is supported, returning all zeros for {country_code}")
            return pd.Series([0] * len(dates), index=dates.index)
        
        try:
            # Convert to date objects if needed
            if dates.dtype == 'object' or 'datetime' not in str(dates.dtype):
                date_series = pd.to_datetime(dates)
            else:
                date_series = dates
            
            # Extract just the date part
            dates_only = date_series.dt.date
            
            # Vectorized lookup using isin
            return dates_only.isin(self.norwegian_holiday_dates).astype(int)
            
        except Exception as e:
            logger.error(f"Error in vectorized holiday lookup: {e}")
            return pd.Series([0] * len(dates), index=dates.index)
    
    def get_holiday_season_type(self, date, country_code: str) -> str:
        """
        Determine the type of holiday season.
        
        Returns: 'summer', 'winter', 'spring', 'easter', 'none'
        """
        try:
            if isinstance(date, str):
                date_obj = datetime.datetime.strptime(date, '%Y-%m-%d').date()
            elif hasattr(date, 'date'):
                date_obj = date.date()
            else:
                date_obj = pd.to_datetime(date).date()
            
            month = date_obj.month
            
            # If not a vacation day, return 'none'
            if not self.is_vacation_day(date, country_code):
                return 'none'
            
            # Classify by month
            if month in [6, 7, 8]:
                return 'summer'
            elif month in [12, 1]:
                return 'winter'
            elif month in [3, 4]:
                return 'easter'
            elif month in [2, 10, 11]:
                return 'autumn'
            else:
                return 'spring'
                
        except Exception as e:
            logger.error(f"Error in get_holiday_season_type: {e}")
            return 'none'
    
    def days_to_holiday(self, date, country_code: str, max_days: int = 14) -> int:
        """
        Calculate days until next holiday period starts.
        
        Returns:
            Number of days until next holiday (0 if already in holiday, -1 if no holiday within max_days)
        """
        try:
            if isinstance(date, str):
                date_obj = datetime.datetime.strptime(date, '%Y-%m-%d').date()
            elif hasattr(date, 'date'):
                date_obj = date.date()
            else:
                date_obj = pd.to_datetime(date).date()
            
            # If already in holiday, return 0
            if self.is_vacation_day(date, country_code, buffer_days=0):
                return 0
            
            # Check next max_days
            for i in range(1, max_days + 1):
                check_date = date_obj + datetime.timedelta(days=i)
                if self.is_vacation_day(check_date.strftime('%Y-%m-%d'), country_code, buffer_days=0):
                    return i
            
            return -1  # No holiday within max_days
            
        except Exception as e:
            logger.error(f"Error in days_to_holiday: {e}")
            return -1
    
    def add_holiday_features(self, df: pd.DataFrame, date_col: str = 'std', 
                           origin_country: str = 'norway', dest_country: str = 'norway') -> pd.DataFrame:
        """
        Add comprehensive holiday features to flight dataset using optimized vectorized operations.
        
        Args:
            df: DataFrame with flight data
            date_col: Name of the date column
            origin_country: Origin country for holiday calculation
            dest_country: Destination country for holiday calculation
        
        Returns:
            DataFrame with added holiday features
        """
        df_result = df.copy()
        
        logger.info(f"Adding holiday features for {len(df)} flights using vectorized operations...")
        
        # Ensure date column is datetime
        if df_result[date_col].dtype == 'object':
            df_result[date_col] = pd.to_datetime(df_result[date_col])
        
        # VECTORIZED: Basic holiday features (much faster)
        logger.info("Adding basic vacation indicators (vectorized)...")
        df_result['is_vacation_origin'] = self.is_vacation_day_vectorized(df_result[date_col], origin_country)
        df_result['is_vacation_dest'] = self.is_vacation_day_vectorized(df_result[date_col], dest_country)
        
        # Combined vacation indicator
        df_result['is_vacation_any'] = (df_result['is_vacation_origin'] | df_result['is_vacation_dest']).astype(int)
        df_result['is_vacation_both'] = (df_result['is_vacation_origin'] & df_result['is_vacation_dest']).astype(int)
        
        # VECTORIZED: Holiday season types
        logger.info("Adding holiday season types (vectorized)...")
        df_result['holiday_season_origin'] = self.get_holiday_season_type_vectorized(df_result[date_col], origin_country)
        df_result['holiday_season_dest'] = self.get_holiday_season_type_vectorized(df_result[date_col], dest_country)
        
        # VECTORIZED: Days to next holiday
        logger.info("Adding days to holiday (vectorized)...")
        df_result['days_to_holiday_origin'] = self.days_to_holiday_vectorized(df_result[date_col], origin_country)
        df_result['days_to_holiday_dest'] = self.days_to_holiday_vectorized(df_result[date_col], dest_country)
        
        # Pre/post holiday indicators (travel surge periods)
        df_result['pre_holiday'] = (df_result['days_to_holiday_origin'].between(1, 3) | 
                                   df_result['days_to_holiday_dest'].between(1, 3)).astype(int)
        
        # VECTORIZED: Holiday intensity
        logger.info("Calculating holiday intensity (vectorized)...")
        df_result['holiday_intensity_origin'] = self.holiday_intensity_vectorized(df_result[date_col], origin_country)
        df_result['holiday_intensity_dest'] = self.holiday_intensity_vectorized(df_result[date_col], dest_country)
        
        # VECTORIZED: Peak travel days
        df_result['is_peak_travel_day'] = self.identify_peak_travel_days_vectorized(df_result[date_col], origin_country)
        
        logger.info(f"Holiday features added successfully! New shape: {df_result.shape}")
        
        # Print feature summary
        holiday_features = [col for col in df_result.columns if 'vacation' in col or 'holiday' in col]
        logger.info(f"Added {len(holiday_features)} holiday features: {holiday_features}")
        
        return df_result
    
    def get_holiday_season_type_vectorized(self, dates: pd.Series, country: str = 'norway') -> pd.Series:
        """Vectorized version of holiday season type detection."""
        try:
            # Default to 'none'
            result = pd.Series(['none'] * len(dates), index=dates.index)
            
            # Get month for each date
            months = pd.to_datetime(dates).dt.month
            
            # Apply season rules based on month
            result.loc[months.isin([6, 7, 8])] = 'summer'
            result.loc[months.isin([12, 1])] = 'winter' 
            result.loc[months.isin([3, 4])] = 'easter'
            result.loc[months.isin([2])] = 'winter'
            result.loc[months.isin([10, 11])] = 'autumn'
            
            # Only set season if it's actually a holiday
            is_holiday = self.is_vacation_day_vectorized(dates, country)
            result.loc[is_holiday == 0] = 'none'
            
            return result
            
        except Exception as e:
            logger.error(f"Error in vectorized season type: {e}")
            return pd.Series(['none'] * len(dates), index=dates.index)
    
    def days_to_holiday_vectorized(self, dates: pd.Series, country: str = 'norway', max_days: int = 14) -> pd.Series:
        """Vectorized version of days to holiday calculation."""
        try:
            result = pd.Series([-1] * len(dates), index=dates.index)
            
            # For each date, check if it's already a holiday
            is_holiday = self.is_vacation_day_vectorized(dates, country)
            result.loc[is_holiday == 1] = 0
            
            # For non-holidays, find days to next holiday (simplified)
            # This is approximate but much faster
            date_objects = pd.to_datetime(dates).dt.date
            
            for i in range(len(dates)):
                if result.iloc[i] != 0:  # Not already a holiday
                    current_date = date_objects.iloc[i]
                    for days_ahead in range(1, max_days + 1):
                        check_date = current_date + datetime.timedelta(days=days_ahead)
                        if check_date in self.norwegian_holiday_dates:
                            result.iloc[i] = days_ahead
                            break
            
            return result
            
        except Exception as e:
            logger.error(f"Error in vectorized days to holiday: {e}")
            return pd.Series([-1] * len(dates), index=dates.index)
    
    def holiday_intensity_vectorized(self, dates: pd.Series, country: str = 'norway', window: int = 7) -> pd.Series:
        """Vectorized version of holiday intensity calculation."""
        try:
            result = pd.Series([0.0] * len(dates), index=dates.index)
            
            date_objects = pd.to_datetime(dates).dt.date
            
            for i in range(len(dates)):
                current_date = date_objects.iloc[i]
                intensity = 0
                
                for j in range(window):
                    check_date = current_date + datetime.timedelta(days=j)
                    if check_date in self.norwegian_holiday_dates:
                        intensity += 1
                
                result.iloc[i] = intensity / window
            
            return result
            
        except Exception as e:
            logger.error(f"Error in vectorized holiday intensity: {e}")
            return pd.Series([0.0] * len(dates), index=dates.index)
    
    def identify_peak_travel_days_vectorized(self, dates: pd.Series, country: str = 'norway') -> pd.Series:
        """Vectorized version of peak travel day identification."""
        try:
            result = pd.Series([0] * len(dates), index=dates.index)
            
            date_objects = pd.to_datetime(dates).dt.date
            
            for i in range(len(dates)):
                current_date = date_objects.iloc[i]
                prev_date = current_date - datetime.timedelta(days=1)
                
                is_holiday_today = current_date in self.norwegian_holiday_dates
                is_holiday_prev = prev_date in self.norwegian_holiday_dates
                
                # Peak travel day if transition into or out of holiday
                if (is_holiday_today and not is_holiday_prev) or (not is_holiday_today and is_holiday_prev):
                    result.iloc[i] = 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error in vectorized peak travel days: {e}")
            return pd.Series([0] * len(dates), index=dates.index)
    
    def _identify_peak_travel_days(self, df: pd.DataFrame, date_col: str, country: str) -> pd.Series:
        """Legacy method - use identify_peak_travel_days_vectorized instead."""
        return self.identify_peak_travel_days_vectorized(df[date_col], country)

# Example usage functions
def create_norwegian_holiday_features(df: pd.DataFrame, date_col: str = 'std') -> pd.DataFrame:
    """
    Convenience function to add Norwegian holiday features to flight data.
    Since most flights will be Norway-related.
    """
    holiday_engineer = HolidayFeatureEngineer()
    return holiday_engineer.add_holiday_features(
        df, 
        date_col=date_col, 
        origin_country='norway', 
        dest_country='norway'
    )

def get_holiday_feature_list() -> List[str]:
    """Return list of holiday feature column names that will be created."""
    return [
        'is_vacation_origin',
        'is_vacation_dest', 
        'is_vacation_any',
        'is_vacation_both',
        'holiday_season_origin',
        'holiday_season_dest',
        'days_to_holiday_origin',
        'days_to_holiday_dest',
        'pre_holiday',
        'holiday_intensity_origin',
        'holiday_intensity_dest',
        'is_peak_travel_day'
    ]