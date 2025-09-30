# H2O Regression Pipeline for Flight Time Prediction

Note that you need to generate the engineered dataset in src/feature_engineering in order to run this notebook. 

This document provides an overview of the `h2o_regression_pipeline2.ipynb` notebook, which implements a regression pipeline using H2O AutoML to predict actual departure and arrival times for flights.

## Objective
The primary goal of this notebook is to predict the exact timing of flights, focusing on:
- **Departure Time (ATD)**: Actual Time of Departure.
- **Arrival Time (ATA)**: Actual Time of Arrival.

## Key Features
- **Multi-Target Regression**: Predicts both departure and arrival times.
- **Evaluation Metrics**: Uses RMSE, MAE, and R² to evaluate model performance.
- **Concurrency Analysis**: Calculates concurrency metrics based on actual and predicted times.

## Workflow

### 1. Import Required Libraries
The notebook begins by importing necessary libraries, including:
- `h2o` for AutoML.
- `pandas` and `numpy` for data manipulation.
- `matplotlib` and `seaborn` for visualization.
- `sklearn` for additional metrics.

### 2. Data Loading
- Historical flight data is loaded from `flight_weather_features.csv`.
- The dataset includes features such as scheduled times, weather conditions, and airport information.

### 3. Data Preprocessing
- **Feature Removal**: Unnecessary columns (e.g., raw time columns) are dropped.
- **Datetime Conversion**: Scheduled and actual times are converted to datetime format.
- **Target Variables**: Regression targets are created by calculating the deviation from scheduled times.
- **Handling Missing Data**: Rows with missing actual times are removed.

### 4. Feature Engineering
- **Temporal Features**: Extracts features such as hour, day of the week, and month.
- **Circular Encoding**: Applies sine and cosine transformations to time-based features.
- **Weather Features**: Merges weather data with flight data, including temperature, humidity, and wind speed.
- **Concurrency Features**: Computes concurrency metrics to capture overlapping flight times at airports.
- **Holiday Features**: Adds indicators for vacation seasons, summer/winter breaks, and Easter periods using OpenHolidaysAPI.
- **Flight Distance**: Calculates the great-circle distance between origin and destination airports using the Haversine formula.

### 5. Train-Test Split
- The dataset is split into training, validation, and test sets.
- The test set is based on October 2024 data to simulate future predictions.

### 6. Model Training
- **H2O AutoML**: Trains separate models for departure and arrival times.
  - Algorithms include XGBoost, GBM, and Deep Learning.
  - Models are evaluated using validation data.
- **Concurrency Metrics**: Calculates concurrency based on actual and predicted times.

### 7. Model Evaluation
- **Performance Metrics**: Evaluates models on the test set using RMSE, MAE, and R².
- **Leaderboard**: Displays the top-performing models for both targets.
- **Variable Importance**: Visualizes the most important features for each model.

### 8. Predictions
- Predicts departure and arrival delays for the test set.
- Saves predictions to `predicted_schedule_october.csv`.

### 9. Concurrency Analysis
- Calculates concurrency metrics for actual and predicted times.
- Compares concurrency levels to identify patterns and discrepancies.

## Outputs
- **Predictions**: Saved as `predicted_schedule_october.csv`.
- **Model Artifacts**: Best models are saved for future use.
- **Visualizations**: Includes variable importance plots and prediction comparisons.

