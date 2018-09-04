"""
this module handles all data clearning and additional features
"""

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = 'data/train.csv'
TRAIN_FEATHER = 'data/train.feather'
TEST_PATH = 'data/test.csv'
SUBMISSION = 'submission.csv'
FEATHER_THREAD = 4


def read(csv_file, nrows = None, train_set=True):
    """
    read csv file with pandas
    """
    traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

    testtypes = {'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

    traincols = list(traintypes.keys())
    testcols = list(testtypes.keys())

    if train_set:
        return pd.read_csv(csv_file, nrows = nrows, usecols = traincols, dtype = traintypes)
    else:
        return pd.read_csv(csv_file, nrows = nrows, usecols = testcols, dtype = testtypes)

def csv2feather(nrows = None):
    """
    read the train csv file and save as feather

    Args:
        nrows: number of rows to read
    """
    traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}
    traincols = list(traintypes.keys())
    df = pd.read_csv(TRAIN_PATH, nrows = nrows, usecols = traincols, dtype = traintypes)
    df.to_feather(TRAIN_FEATHER)

def clean(df, train_set=True):
    """
    with pandas, this cleans the original rows, deleteling NAs, 
    and very unreasonable rows

    right now only support train set 
    """
    if train_set:
        df = df.dropna(how='any', axis='rows') ##drop NAs
        df = df[df.fare_amount>=0] ## drop negative fare

        ##drop latitude and longitude that are not in NYC
        idx = (df.pickup_longitude<-70) & (df.pickup_longitude>-77) & \
              (df.dropoff_longitude<-70) & (df.dropoff_longitude>-77) & \
              (df.pickup_latitude<45) & (df.pickup_latitude>37) & \
              (df.dropoff_latitude<45) & (df.dropoff_latitude>37)

        return df[idx]

    else:
        pass
       

def add_features(df):
    """
    adds more features
    """
    ##add vertical distance and horizontal distance
    df['delta_lon'] = df.pickup_longitude - df.dropoff_longitude
    df['delta_lat'] = df.pickup_latitude - df.dropoff_latitude

    ##parse datetime (Note: read_csv.parse_datatime too too slow)
    df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 16)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    df['year'] = df.pickup_datetime.apply(lambda x:x.year)
    df['weekday'] = df.pickup_datetime.apply(lambda x:x.weekday())
    df['hour'] = df.pickup_datetime.apply(lambda x:x.hour)

    ##one_hot encoding
    #df = pd.get_dummies(df, columns=['weekday'])

    return df

def df_to_matrix(df, train_set=True):
    """
    pick features to numpy matrix
    """
    features = ['passenger_count', 'year', 'weekday', 'hour', 'delta_lon', 'delta_lat']
    X = df[features].values
    if train_set:
        y = df['fare_amount'].values
        return X, y
    return X

def input(train_row=None):
    """
    normalize the features, provide inputs
    """
    print('read training data')
    #df = read(TRAIN_PATH, nrows = train_row)
    df = pd.read_feather(TRAIN_FEATHER, nthreads=FEATHER_THREAD)
    df = clean(df)
    df = add_features(df)
    X_train, y_train =  df_to_matrix(df)

    print('read test data')
    df = read(TEST_PATH, train_set=False)
    #df = clean(df, train_set=False)
    df = add_features(df)
    X_test = df_to_matrix(df, train_set=False)

    print('normalize data')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print('data read complete')

    return X_train, y_train, X_test

if __name__=='__main__':
    # ##read train set, small test
    start = time.time()
    #------------------test 1----------------------
    # df_train = read(TRAIN_PATH, nrows = 100000)
    # print('Orignial size : {}'.format(len(df_train)))
    # df_train = clean(df_train)
    # print('Cleaned size : {}'.format(len(df_train)))
    # df_train = add_features(df_train)
    # print(df_train.head(5))
    # X, y = df_to_matrix(df_train, train_set=True)
    # print(X.shape)
    # print(y.shape)
    # print(X[:5])
    # print(y[:5])
    #--------------------test 2---------------------
    #csv2feather()
    #-------------------test 3----------------------
    X_train, y_train, X_test = input()
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(X_train[:5])
    print(y_train[:5])
    print(X_test[:5])
    #--------------------------------------------
    end = time.time()
    print("Time spent {}".format(end - start))



    





