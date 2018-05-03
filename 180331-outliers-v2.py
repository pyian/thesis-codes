"""
Remove outliers
v2: also plot which data points removed by which outlier detection method

NOTE: ADDED TRAFFIC DATA, ASSUME THEY DON'T HAVE OUTLIERS
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import dates
sns.set_style('whitegrid')
sns.set_context("poster")


# Global Variables
threshold = 40
threshold_zsc = 3
win_hour = 120
win_day = win_hour * 24
win_week = win_day * 7
win_month = win_day * 30
path = 'frb_all_zdcprd.h5'


def main(df):
    """do for each column"""
    dfs = pd.DataFrame()
    for col in df.columns:
        print('Processing {}'.format(col))
        dfs = pd.concat([dfs, na_outlier(df[col], col)], axis=1)

    print('Finished Processing')
    dfs.to_hdf('frb_all_zdcprdo.h5', 'table', data_columns=True)
    print('WROTE TO frb_all_zdcprdo.h5')

    # for sensor in dfs.columns:
    # fig_name = "sensor {}".format(sensor)
    # plot_rolling(dfs[sensor], win_week, win_day * 5, fig_name, 600)


def na_outlier(series, col):
    d = pd.DataFrame(series)
    plt.figure(dpi=600)
    plt.plot(d[col], label='Data', linewidth=1)
    # identify outliers by rolling method
    d = det_median(d, col)
    # identify outliers by zscore
    d = det_zscore(d, col)

    plt.legend()
    fig_title = 'sensor {}'.format(col)
    plt.title(fig_title)
    fig_name = fig_title.replace(" ", "_") + '_id.png'
    plt.savefig('sen_figs/' + fig_name)
    print("NEW FIGURE: {}".format(fig_name))
    plt.close('all')

    # remove outliers
    d.loc[d['median_diff'] >= threshold, col] = np.nan
    d.loc[d['zscore'] >= threshold_zsc, col] = np.nan

    return d[col]


def det_median(d, col):
    # remove outliers by rolling method
    d['median'] = d[col].rolling(window=win_hour).median()
    d['median_diff'] = (d[col] - d['median']).abs()
    try:
        plt.plot(d.loc[d['median_diff'] >= threshold, col], marker='.', linewidth=0,
                 c='orange', label='Median Filter', alpha=0.4, markersize=7)
    except:
        pass
    return d


def det_zscore(d, col):
    d['zscore'] = (d[col] - d[col].mean()) / d[col].std()
    d['zscore'] = d['zscore'].abs()
    try:
        plt.plot(d.loc[d['zscore'] >= threshold_zsc, col], marker='.',
                 linewidth=0, c='red', label='Zscore Filter', alpha=0.4, markersize=7)
    except:
        pass
    return d


def plot_rolling(df, window, min_periods, fig_title, fig_dpi):
    """Plot Rolling median and std

    Keyword arguments:
    df -- pd.series
    window -- size of window (according to the array)
    min_periods -- minimum periods for it to plot
    """
    rolmean = df.rolling(window=window, min_periods=min_periods).mean()
    rolmedian = df.rolling(window=window, min_periods=min_periods).median()
    # rolstd = df.rolling(window=window, min_periods=min_periods).std()

    fig, ax1 = plt.subplots()

    ax1.plot(df, label='Data', linewidth=1)
    ax1.plot(rolmean, label='Rolling Mean', linewidth=1)
    ax1.plot(rolmedian, label='Rolling Median', linewidth=1)
    ax1.set_title(fig_title)
    ax1.legend()
    ax1.xaxis.set_major_formatter(dates.DateFormatter("%m/%y"))
    plt.xticks(rotation=90)

    fig_name = fig_title.replace(" ", "_") + '_out3.png'
    plt.savefig('sen_figs/' + fig_name)
    print("NEW FIGURE: {}".format(fig_name))
    plt.close('all')


if __name__ == '__main__':
    df = pd.read_hdf(path, 'table')
    main(df)
