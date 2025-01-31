/*
Data Preparation
Author: Lukas Coffey
Date: 2025-01-07

Overview
This notebook contains the feature engineering steps for the Load Forecasting Model project.

Dependencies
Python libraries: pandas, numpy, matplotlib
Databricks cluster configured with Python 3.x
*/

/*
Create Training Table
Setting up required table to contain model training features.
Note: For brevity, weather data will be added later since it contains 54 additional features.
*/

%sql
CREATE OR REPLACE TABLE dev.load_forecast_db_gold.load_training (
  interval_start_local TIMESTAMP COMMENT 'Timestamp of start of current interval in local time.',
  interval_start_utc TIMESTAMP COMMENT 'Timestamp of start of current interval',
  interval_end_utc TIMESTAMP COMMENT 'Timestamp of end of current interval',
  day_of_week STRING COMMENT 'Day of week corresponding to local time.',
  holiday BOOLEAN COMMENT 'Holiday corresponding to local time.',
  weekend BOOLEAN COMMENT 'Weekend flag based on day_of_week.',
  season STRING COMMENT 'Season corresponding to UTC time.',
  native_load DOUBLE COMMENT 'The native load at interval start.'
);
