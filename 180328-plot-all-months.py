import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sys import argv

sns.set_style('white')
sns.set_context("poster")

"""
Plot all sensors by month
Window sizes:
    1440: 12 hours
    2880: 24 hours
    20160: 1 week
Config:
    Rolling window: 1 week
    Minimum Period: 5 days

"""

win_hour = 120
win_day = win_hour * 24
win_week = win_day * 7
win_month = win_day * 30


def read_data(path):
    df = pd.read_hdf(path, 'table')
    return df


def plot_all_sensors(df):
    sensors = df.columns
    for sensor in sensors:
        fig_name = "{} sensor {}".format(path[4:11], sensor)
        plot_rolling(df[sensor], win_day, win_hour * 12, fig_name, 600)
    print('Script Finished')


def plot_rolling(df, window, min_periods, fig_title, fig_dpi):
    """Plot Rolling median and std
    input
    df: dataframe
    window: size of window (according to the array)
    min_periods: minimum periods for it to plot
    """
    rolmean = df.rolling(window=window, min_periods=min_periods).mean()
    rolmedian = df.rolling(window=window, min_periods=min_periods).median()
    rolstd = df.rolling(window=window, min_periods=min_periods).std()
    plt.figure(dpi=fig_dpi)
    plt.plot(df, label='Data', linewidth=1)
    plt.plot(rolmean, label='Rolling Mean', linewidth=1)
    plt.plot(rolmedian, label='Rolling Median', linewidth=1)
    plt.plot(rolstd, label='Rolling STD', linewidth=1)
    plt.legend()
    plt.title(fig_title)
    fig_name = 'fig-' + fig_title.replace(" ", "_") + '.png'
    plt.savefig('sen_figs/' + fig_name)
    print("NEW FIGURE: {}".format(fig_name))
    plt.close('all')


def main(path):
    df = read_data(path)
    plot_all_sensors(df)
    print('Script Finished')


if __name__ == '__main__':
    path = argv[1]
    main(path)
