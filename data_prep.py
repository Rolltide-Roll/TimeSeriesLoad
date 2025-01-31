###
#Data Preparation
#Author: Lukas Coffey
#Date: 2025-01-07

#Overview
#This notebook contains the feature engineering steps for the Load Forecasting Model project.

#Dependencies
#Python libraries: pandas, numpy, matplotlib
#Databricks cluster configured with Python 3.x
###

'''
Create Training Table
Setting up required table to contain model training features.
Note: For brevity, weather data will be added later since it contains 54 additional features.
'''

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

spark.sql("SELECT * FROM dev.load_forecast_db_gold.load_training").columns

'''
Feature Engineering
Dates
We will use the dates table to create a baseline of every date in our training period. We will then join data into those dates and use this to find any missing or duplicate values.

We only want hourly data between January 1st 2022 and December 31st 2024.
'''

dates = spark.sql("""
                  SELECT 
                    datetime_local as interval_start_local,
                    datetime_utc AS interval_start_utc,
                    DATE_ADD(HOUR, 1, datetime_utc) AS interval_end_utc,
                    day_of_week
                  FROM dev.load_forecast_db_gold.dates
                  WHERE 
                    year >= 2022
                    AND year <= 2024
                    AND minute = 0
                  ORDER BY datetime_utc
                  """)

# Check the start and end dates
spark.createDataFrame(dates.head(30)).display()
spark.createDataFrame(dates.tail(30)).display()

'''
Holidays
We are adding in U.S. holidays for the training period only, later on this process can be automated. These holidays will be joined using the local time, not UTC time, 
since the holiday feature needs to coincide with the actual local celebration times.
'''

from datetime import date

# Dictionary to store traditional US holidays for the last 3 years
us_holidays = [
    # 2022:
    date(2022, 1, 1),  # New Year's Day
    date(2022, 1, 17),  # Martin Luther King Jr. Day
    date(2022, 2, 21),  # Presidents' Day
    date(2022, 5, 30),  # Memorial Day
    date(2022, 7, 4),  # Independence Day
    date(2022, 9, 5),  # Labor Day
    date(2022, 10, 10),  # Columbus Day
    date(2022, 11, 11),  # Veterans Day
    date(2022, 11, 24),  # Thanksgiving Day
    date(2022, 12, 25),  # Christmas Day
    # 2023:
    date(2023, 1, 1),  # New Year's Day
    date(2023, 1, 16),  # Martin Luther King Jr. Day
    date(2023, 2, 20),  # Presidents' Day
    date(2023, 5, 29),  # Memorial Day
    date(2023, 7, 4),  # Independence Day
    date(2023, 9, 4),  # Labor Day
    date(2023, 10, 9),  # Columbus Day
    date(2023, 11, 11),  # Veterans Day
    date(2023, 11, 23),  # Thanksgiving Day
    date(2023, 12, 25),  # Christmas Day
    # 2024:
    date(2024, 1, 1),  # New Year's Day
    date(2024, 1, 15),  # Martin Luther King Jr. Day
    date(2024, 2, 19),  # Presidents' Day
    date(2024, 5, 27),  # Memorial Day
    date(2024, 7, 4),  # Independence Day
    date(2024, 9, 2),  # Labor Day
    date(2024, 10, 14),  # Columbus Day
    date(2024, 11, 11),  # Veterans Day
    date(2024, 11, 28),  # Thanksgiving Day
    date(2024, 12, 25)  # Christmas Day
]

from pyspark.sql.functions import to_date

# Join dates to us_holidays as a boolean flag
dates = dates.withColumn('holiday', to_date(dates.interval_start_local).isin(us_holidays))

# Check that all holidays are joined
dates.filter(dates.holiday == True).display()

'''
Weekends
Add a flag for weekend/not weekend.
'''

from pyspark.sql.functions import when

dates = dates.withColumn('weekend', when(dates.day_of_week.isin(['Sat', 'Sun']), True).otherwise(False))
spark.createDataFrame(dates.tail(30)).display()

'''
Seasonal Strata & Timezones
Adding logic to handle daylight savings which will also be used to flag two seasons, winter and summer. This will then be used in the model to create the combined holiday/weekend strata.

Timezone and Daylight Savings Methodology
Daylight savings causes 1 timestamp per year to be repeated. We want to know when this happens so we can properly mark the season change later on. We will also use this dst_order value 
to verify data in the load_actuals dataframe.
'''

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

# Creating a window specification for timestamps and ordering by the UTC time (since UTC is by default in order)
window_spec = Window.partitionBy("interval_start_local").orderBy(dates.interval_start_utc)

# We apply the window function, and verify that on DST days there are row numbers 1 and 2 for repeated timestamps
dates = dates.withColumn("dst_order", row_number().over(window_spec))
dates.filter(to_date(dates.interval_start_local) == '2022-11-6').display()

from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
from datetime import datetime

dst_changes = {
    2022: {
        "start": datetime(2022, 3, 13, 2, 0),
        "end": datetime(2022, 11, 6, 2, 0),
    },
    2023: {
        "start": datetime(2023, 3, 12, 2, 0),
        "end": datetime(2023, 11, 5, 2, 0),
    },
    2024: {
        "start": datetime(2024, 3, 10, 2, 0),
        "end": datetime(2024, 11, 3, 2, 0),
    },
}

# Define UDF to classify timestamps as EST/EDT
def flag_dst_udf(timestamp, dst_order):
    year = timestamp.year
    
    if year in dst_changes:
        dst_start = dst_changes[year]["start"]
        dst_end = dst_changes[year]["end"]

        if dst_order == 2:
            return "Winter"
        elif dst_start < timestamp < dst_end:
            return "Summer"
        else:
            return "Winter"
    else:
        return None
    
# Convert Python function to a PySpark UDF
flag_dst_spark_udf = udf(flag_dst_udf, StringType())

# Call the function on each datetime, then verify the change on a DST day
dates = dates.withColumn("season", flag_dst_spark_udf(dates.interval_start_local, dates.dst_order))
dates.filter(to_date(dates.interval_start_local) == '2022-11-6').display()

'''
Load Feature
We add the data from the PCI load table. The load actuals are recorded as "hour end" values, meaning the value in the NATIVE column is the value at the end of an hour interval. 
We need to offset these values since weather data is "hour start."

In the query below we are calculating the UTC time of the timestamps and specifying the interval the load value is for. The load value is at "hour end", which means it is also 
the load value for the "hour start" of the next hour. To do this we add 1 hour to the interval timestamps which effectively shifts the load values from "hour end" to "hour start", and the 
interval timestamps now represent the very next interval.
'''

# Selecting native load for the 2022-2024 period
load_actuals = spark.sql("""
    WITH LOAD AS (
        SELECT
            IFF(read_time_zone = 'EDT', 'UTC-4', 'UTC-5') AS offset,
            TO_TIMESTAMP(concat(read_date, ' ', read_hour - 1, ':00:00'), 'yyyy-MM-dd H:mm:ss') AS interval_start,
            DATE_ADD(HOUR, 1, interval_start) AS interval_end,
            TO_UTC_TIMESTAMP(interval_start, offset) AS interval_start_utc,
            TO_UTC_TIMESTAMP(interval_end, offset) AS interval_end_utc,
            native AS native_load_at_end,
            -- Adding a row number to verify the joins and data shift later on
            ROW_NUMBER() OVER (PARTITION BY read_date, read_hour ORDER BY read_hour, read_time_zone) AS dst_order
        FROM 
            dev.pci_db_base.load_actuals_historical
        WHERE read_date > '2022-01-01'
            AND read_date < '2024-12-31'
        ORDER BY read_date, read_hour, read_time_zone
    )
    SELECT 
        interval_start_utc,
        interval_end_utc,
        native_load_at_end as native_load_at_start,
        dst_order -- Keeping this to verify joins and data shift
    FROM load
"""
)

load_actuals.filter(to_date(load_actuals.adj_interval_start_utc) == '2024-11-03').display()

'''
Joining Load Actuals into Dates Frame
Here we merge the load and dates datasets. To verify load data is shifted one hour later (to transform the native_load_at_end value to native_load_at_start) we compare the dst_order columns.
'''

# We define a join condition to specify the mismatching column names
join_condition = (dates.interval_start_utc == load_actuals.interval_start_utc)

# Join and verify
training_data_full = dates.join(load_actuals, on=join_condition, how='left')
training_data_full.filter(to_date(col('interval_start_utc')) == '2024-11-03').display()

'''
Feature Trimming & Initial Load
After verifying the join above, we select only the features we want from the new frame to match the load_training table. We then load the data into load_training.
'''

# Select only relevant features
training_data = training_data_full.select(
  col("interval_start_local"),
  col("interval_start_utc"),
  col("interval_end_utc"),
  col("day_of_week"),
  col("holiday"),
  col("weekend"),
  col("season"),
  col("native_load_at_start").alias("native_load"),
)

# Verify
training_data.display()

# Load data into training table
training_data.write.mode("append").insertInto("dev.load_forecast_db_gold.load_training")

'''
Calculating Weighted Coordinates
Using load delivery points and their point-in-time load values to calculate a dataframe of weighted weather stations. We'll use these weather stations in the training data.
'''

load_coords = spark.sql("""
                        SELECT *
                        FROM dev.load_forecast_db_gold.weighted_load_data_with_coordinates
                        """)
load_coords.display()

weather_coords = spark.sql("""
                           SELECT distinct city_id, city_name, lat, lon
                           FROM dev.weatherbit_db_base.historical
                           """)
weather_coords.display()

'''
Haversine Distance Formula
We are using the Haversine formula to calculate the distance between each delivery point and each weather station to find the closest weather station.
'''

from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points 
    on the Earth using the Haversine formula.
    """
    R = 6371  # Radius of Earth in km
    lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c  # Distance in km

def find_closest_location(target_lat, target_lon, locations):
    """
    Finds the closest latitude/longitude pair to a given target.
    
    :param target_lat: Target latitude (float)
    :param target_lon: Target longitude (float)
    :param locations: List of (latitude, longitude) tuples
    :return: Closest (latitude, longitude) tuple
    """
    return min(locations, key=lambda loc: haversine(target_lat, target_lon, loc['lat'], loc['lon']))

weighted_points_arr = []
for point in load_coords.collect():
    locations = weather_coords.collect()  # (NYC, LA, London)
    target = (point['LAT'], point['LONG'])  # Philadelphia
    closest_station = find_closest_location(target[0], target[1], locations)
    print(f"Delivery Point: {point['GIS Name']}, Station: {closest_station['city_id']}")
    weighted_points_arr.append({
        'closest_station': closest_station,
        'name': point['GIS Name'],
        'lat': point['LAT'],
        'lng': point['LONG'],
        'PLoad': point['Pload (MW)']
    })

weighted_points = spark.createDataFrame(weighted_points_arr)
weighted_points.display()

# Reformatting the output into full columns
weighted_points = weighted_points.selectExpr(
    "PLoad",
    "closest_station.city_id AS closest_station_id",
    "closest_station.city_name AS closest_station_name",
    "closest_station.lat AS closest_station_lat",
    "closest_station.lon AS closest_station_lon",
    "lat",
    "lng",
    "name"
)
weighted_points.display()

weighted_points.createOrReplaceTempView("weighted_points")

spark.sql("""
           SELECT 
            closest_station_name,
            SUM(PLoad) as weighted_load,
            SUM(weighted_load)
           FROM weighted_points
           GROUP BY closest_station_name
           ORDER BY weighted_load desc
           """).display()














