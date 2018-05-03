"""
Compare different models

Feature:
    traffic
    wind speed
    wind direction
    tmperature
Target:
    strain
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Dropout
from keras.layers import LSTM

sns.set_style('white')
sns.set_context("poster")


def gen_shift(d, n):
    """from 1 to n"""
    dds = []

    for i in np.array([1, 2, 480, 960, 1440, 2880]):
        dd = d.shift(i).drop(['str_245'], axis=1)
        cols = dd.columns
        cols = change_col_names(dd, cols, i)
        dd.columns = cols
        dds.append(dd)

    for ddd in dds:
        d = pd.concat([d, ddd])

    return d


def change_col_names(dd, cols, i):
    new_names = []

    for name in dd.columns:
        name += "_{}".format(i)
        new_names.append(name)

    return new_names


def feature_scaling(df):
    scaler = MinMaxScaler()

    df[['trf_count', 'weather_9001', 'weather_9011', 'weather_9002',
        'weather_9012', 'str_245', 'tmp_269']] = scaler.fit_transform(df[['trf_count', 'weather_9001', 'weather_9011', 'weather_9002', 'weather_9012', 'str_245', 'tmp_269']])
    return df


def print_r2mae(r2s, maes):
    print("----------RESULTS----------")
    print("R2 Scores: {}".format(r2s))
    print("R2 Mean: {:.4f}".format(np.mean(r2s)))
    print("R2 STD: {:.4f}".format(np.std(r2s)))
    print("MAE: {}".format(maes))
    print("MAE Mean: {:.4f}".format(np.mean(maes)))
    print("MAE STD: {:.4f}".format(np.std(maes)))
    rms = np.sqrt(maes)
    print("RMS: {}".format(rms))
    print("RMS Mean: {:.4f}".format(np.mean(rms)))
    print("RMS STD: {:.4f}".format(np.std(rms)))
    print("---------------------------")


def linear_regression(X_train, X_test, y_train, y_test):
    print('Linear Regression Model')
    r2s = []
    maes = []

    for _ in np.arange(5):
        lr = LinearRegression().fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2s.append(r2)
        mae = mean_absolute_error(y_test, y_pred)
        maes.append(mae)

    print_r2mae(r2s, maes)


def MLPR(X_train, X_test, y_train, y_test):
    print('ANN')
    r2s = []
    maes = []
    for _ in np.arange(5):
        ann = MLPRegressor(hidden_layer_sizes=(16)).fit(X_train, y_train)
        y_pred = ann.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2s.append(r2)
        mae = mean_absolute_error(y_test, y_pred)
        maes.append(mae)

    print_r2mae(r2s, maes)


def Deep(X_train, X_test, y_train, y_test):
    print('Deep Learning')
    r2s = []
    maes = []
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    for _ in np.arange(5):
        # model
        model = Sequential()
        model.add(Dense(16, activation='relu', input_dim=6))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='relu'))
        # model.add(Dense(1, activation='relu'))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=5, batch_size=32)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2s.append(r2)
        mae = mean_absolute_error(y_test, y_pred)
        maes.append(mae)

    print_r2mae(r2s, maes)


def Deep_shift(X_train, X_test, y_train, y_test):
    print('Deep Learning')
    r2s = []
    maes = []
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values
    for _ in np.arange(5):
        # model
        model = Sequential()
        model.add(Dense(16, activation='relu', input_dim=42))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='relu'))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=5, batch_size=32)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2s.append(r2)
        mae = mean_absolute_error(y_test, y_pred)
        maes.append(mae)

    print_r2mae(r2s, maes)


def ann_LSTM(X_train, X_test, y_train, y_test):
    print('LSTM')
    r2s = []
    maes = []
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    for _ in np.arange(5):
        model = Sequential()
        model.add(LSTM(16, input_shape=(1, 42), return_sequences=True))
        model.add(LSTM(8))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        model.fit(X_train, y_train, epochs=5)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2s.append(r2)
        mae = mean_absolute_error(y_test, y_pred)
        maes.append(mae)

    print_r2mae(r2s, maes)


if __name__ == '__main__':

    df = pd.read_hdf('elite_sen_final.h5', 'table')
    df = df.drop(['tmp_211', 'str_172'], axis=1)
    df = df.loc['2017-07-01':'2017-08-29']
    df['str_245'] = df['str_245'].interpolate()
    df['tmp_269'] = df['tmp_269'].interpolate()
    df = feature_scaling(df)

    # before shfit
    X = df.drop('str_245', axis=1)
    y = df.str_245
    X_train = X.loc[:'2017-08-20']
    y_train = y.loc[:'2017-08-20']
    X_test = X.loc['2017-08-20':]
    y_test = y.loc['2017-08-20':]

    # linear_regression(X_train, X_test, y_train, y_test)
    # MLPR(X_train, X_test, y_train, y_test)
    # Deep(X_train, X_test, y_train, y_test)

    # time shift
    df = gen_shift(df, 6).groupby(level=0).sum().dropna()
    print('With time shift:')

    X = df.drop('str_245', axis=1)
    y = df.str_245

    X_train = X.loc[:'2017-08-20']
    y_train = y.loc[:'2017-08-20']

    X_test = X.loc['2017-08-20':]
    y_test = y.loc['2017-08-20':]

    # linear_regression(X_train, X_test, y_train, y_test)
    # MLPR(X_train, X_test, y_train, y_test)
    # Deep_shift(X_train, X_test, y_train, y_test)
    ann_LSTM(X_train, X_test, y_train, y_test)

    print("End of Script.")
