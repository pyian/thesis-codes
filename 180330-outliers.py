"""
Remove outliers
"""

import pandas as pd
import numpy as np
# from scipy import stats
import matplotlib.pyplot as plt

threshold = 50
win_hour = 120
win_day = win_hour * 24
win_week = win_day * 7
win_month = win_day * 30
path = 'frb_all_zdcpr.h5'


def main(df):
    """do for each column"""
    dfs = pd.DataFrame()
    for col in df.columns:
        print('Processing {}'.format(col))
        dfs = dfs.append(process(df[col], col))

    dfs = dfs.T
    print('Finished Processing')
    for sensor in dfs.columns:
        fig_name = "{} sensor {}".format(path[4:11], sensor)
        plot_rolling(dfs[sensor], win_day, win_hour * 12, fig_name, 200)


def process(series, col):
    d = pd.DataFrame(series)
    # remove huge outliers
    d = d.dropna()
    d['median'] = d[col].rolling(window=win_hour).median()
    difference = np.abs(d[col] - d['median'])
    outlier_idx = difference > threshold
    d[col][outlier_idx] = np.nan

    return d[col]


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
    # fig, ax1 = plt.subplots()
    # ax1.plot(df, label='Data', linewidth=1)
    # ax1.plot(rolmean, label='Rolling Mean', linewidth=1)
    # ax1.plot(rolmedian, label='Rolling Median', linewidth=1)
    # ax2 = ax1.twinx()
    # ax2.plot(rolmedian, label='Rolling Median', linewidth=1)
    plt.title(fig_title)
    fig_name = fig_title.replace(" ", "_") + 'fixed.png'
    plt.savefig('sen_figs/' + fig_name)
    print("NEW FIGURE: {}".format(fig_name))


if __name__ == '__main__':
    cols = ['disp_92', 'disp_150', 'disp_254',
            'str_119', 'str_122', 'str_125', 'str_1227']
    df = pd.read_hdf(path, 'table', columns=cols)
    main(df)
