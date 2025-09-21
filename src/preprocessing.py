import os
import pandas as pd
from pandas import DataFrame

class DataPreprocessor:
    def __init__(self):
        pass

    def process_hist_flights(self, df_hist: DataFrame) -> DataFrame:
        """General processing of historical flight data"""
        df_processed_hist = df_hist.copy()

        # Drop rows where cancelled is 1
        df_processed_hist = df_processed_hist[df_processed_hist["cancelled"] == 0]


        std_parsed = pd.to_datetime(df_processed_hist['std']) if 'std' in df_processed_hist.columns else None
        sta_parsed= pd.to_datetime(df_processed_hist['sta']) if 'sta' in df_processed_hist.columns else None
        
        df_processed_hist['std'] = std_parsed
        df_processed_hist['sta'] = sta_parsed

        atd_parsed = pd.to_datetime(df_processed_hist['atd']) if 'atd' in df_processed_hist.columns else None
        ata_parsed = pd.to_datetime(df_processed_hist['ata']) if 'ata' in df_processed_hist.columns else None

        df_processed_hist['atd'] = atd_parsed
        df_processed_hist['ata'] = ata_parsed
        
        # Fill missing service_type values within each flight_id group and keep original format
        df_processed_hist['service_type'] = df_processed_hist.groupby('flight_id')['service_type'].transform(lambda x: x.ffill().bfill())

        #Drop rows where atd and ata are missing, or cancelled is 1
        df_processed_hist = df_processed_hist.dropna(subset=['atd', 'ata'])
        df_processed_hist = df_processed_hist[df_processed_hist['cancelled'] != 1]
        #remove cancelled column
        df_processed_hist = df_processed_hist.drop(columns=['cancelled'])

        print("Missing values after processing:")
        print(df_processed_hist.isna().sum())
        return df_processed_hist

    def process_hist_flights_cancelled(self, df_hist: DataFrame) -> DataFrame:
        """General processing of historical flight data and removal of unused columns"""
        # Historical Flights
        # todo: missing atd and ata:
        # train a model without these values, predict these values, and train a model with these values
        # or completely drop these rows

        df_processed_hist = df_hist.copy()

        std_parsed = pd.to_datetime(df_processed_hist['std']) if 'std' in df_processed_hist.columns else None
        sta_parsed= pd.to_datetime(df_processed_hist['sta']) if 'sta' in df_processed_hist.columns else None
        
        df_processed_hist['std'] = std_parsed
        df_processed_hist['sta'] = sta_parsed

        # Sort by std (scheduled time of departure)
        df_processed_hist = df_processed_hist.sort_values('std')


        #set cancelled to 1 if atd and ata are missing
        df_processed_hist[df_processed_hist['cancelled'].isna() | df_processed_hist['ata'].isna() & df_processed_hist['atd'].isna()]["cancelled"] = 1
        
        #drop actual times for cancellation prediction
        df_processed_hist = df_processed_hist.drop(columns=["atd", "ata"])

        # Fill missing service_type values within each flight_id group and keep original format
        df_processed_hist['service_type'] = df_processed_hist.groupby('flight_id')['service_type'].transform(lambda x: x.ffill().bfill())

        print("Missing values after processing:")
        print(df_processed_hist.isna().sum())


        return df_processed_hist



    def process_future_flights(self, df_fut: DataFrame) -> DataFrame:
        # Implement data cleaning logic here
        df_processed_fut = df_fut.copy()
        return df_processed_fut
    
if __name__ == "__main__":
    df_hist = pd.read_csv(os.path.abspath("data/historical_flights.csv"))
    preprocessor = DataPreprocessor()
    df_processed_hist = preprocessor.process_hist_flights_cancelled(df_hist)
    print(df_processed_hist.head())