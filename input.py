"""
this module handles all data clearning and additional features
"""

import pandas as pd
import numpy as np

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
SUBMISSION = 'submission.csv'

def clean(df):
	"""
	with pandas, this cleans the original rows, deleteling NAs, 
	and very unreasonable rows
	"""
	df = df.dropna(how='any', axis='rows') ##drop NAs
	df = df[df_train.fare_amount>=0] ## drop negative fare

	##drop latitude and longitude that are not in NYC
	df = (-77<df.pickup_longitude<-70) & (-77<df.dropoff_longitude<-70) & \
       (37<df.pickup_latitude<45) & (37<df.dropoff_latitude<45)
    return df
       

def add_features(df):
	"""
	adds more features
	"""
    df['year'] = df.pickup_datetime.apply(lambda x:x.year)
    df['weekday'] = df.pickup_datetime.apply(lambda x:x.weekday())
    df['hour'] = df.pickup_datetime.apply(lambda x:x.hour)

    ##add vertical distance and horizontal distance
    df_train['delta_lon'] = df_train.pickup_longitude - df_train.dropoff_longitude
	df_train['delta_lat'] = df_train.pickup_latitude - df_train.dropoff_latitude
	return df




