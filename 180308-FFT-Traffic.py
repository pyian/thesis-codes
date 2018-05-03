# Plot FFT for traffic data

import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import numpy as np

"""
Apply FFT to traffic data
1. Append all data into a giant df
2. Apply FFT
3. Plot Spectrum
"""


def read_file(path):
    df = pd.read_csv(path)
    # to datetime
    df['timestamp'] = pd.to_datetime(
        df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index('timestamp')
    return df


def apply_FFT(df):
    print(df.columns)
    print(df.head())
    signal = df['volume'].values
    print('Median volume: {:.2f}'.format(np.median(signal)))
    Fs = 1/60
    n = len(df.index)
    print('n: {}'.format(n))
    k = np.arange(n)
    T = n/Fs
    frq = k/T
    frq = frq[range(int(n/2))]
    Y = np.fft.fft(signal)/n
    Y = Y[range(int(n/2))]
    # plot_df(df)
    # plot_FFT(frq, Y)
    # dom_pos = ana_dom_signal(frq, Y)
    return frq, Y


def plot_df(df):
    plt.figure(dpi=300)
    df.plot(ax=plt.gca())
    plt.title('Yearly Traffic Variation')
    plt.savefig('Traffic_Variation')


def plot_FFT(frq, Y):
    # all
    plt.figure(dpi=300)
    plt.plot(frq, abs(Y), 'r')
    plt.xlabel('freq (Hz)')
    plt.ylabel('|Y(freq)|')
    plt.title('Specturm - FULL')
    plt.savefig('Traffic_Spectrum')

    # zoom
    plt.figure(dpi=300)
    plt.plot(frq, abs(Y), 'r')
    plt.xlim(-0.00001, 0.0001)
    plt.xlabel('freq (Hz)')
    plt.ylabel('|Y(freq)|')
    plt.title('Specturm - ZOOMED IN')
    plt.savefig('Traffic_Spectrum_Zoomed')


def ana_dom_signal(frq, A):
    # find dominant frequencies

    # find pos of n most dominant freq
    n = 30
    pos = A.argsort()[-n:][::-1]
    print('{} most dominant frequency (Hz):'.format(n))
    print('Freq (Hz) \t Amp')
    for p in pos:
        print('{:.10f} \t {:.3f}'.format(frq[p], A[p]))
    print('Sum of Y {:.2f}'.format(sum(A)))
    return pos


def gen_signal(df, frq, Y):
    # using 10 most dominant freqs
    A = abs(Y)
    dom_pos = ana_dom_signal(frq, A)
    n = len(df.index)
    t = np.arange(n) * 60
    signal = np.zeros(n)
    for p in dom_pos:
        signal += A[p] * np.sin(2 * np.pi * frq[p] * t)
    return signal


def gen_signal_join_df(df, sig_gen):
    print(len(df.index), len(sig_gen))
    df['gen'] = sig_gen
    print(df.head())
    return df


if __name__ == '__main__':

    # paths = ['tfc-r-2017-1.csv','tfc-r-2017-2.csv', 'tfc-r-2017-3.csv',
    #          'tfc-r-2017-4.csv', 'tfc-r-2017-5.csv', 'tfc-r-2017-6.csv',
    #          'tfc-r-2017-7.csv', 'tfc-r-2017-8.csv', 'tfc-r-2016-10.csv',
    #          'tfc-r-2016-11.csv', 'tfc-r-2016-12.csv']

    paths = ['tfc-r-2017-3.csv']#, 'tfc-r-2017-3.csv',
            # 'tfc-r-2017-4.csv', 'tfc-r-2017-5.csv', 'tfc-r-2017-6.csv',
            # 'tfc-r-2017-7.csv', 'tfc-r-2017-8.csv']
    dfs = []
    dfall = pd.DataFrame()

    for path in paths:
        dfs.append(read_file(path))

    for df in dfs:
        dfall = dfall.append(df)

    dfall = dfall.sort_index()

    frq, Y = apply_FFT(dfall)

    sig_gen = gen_signal(df, frq, Y)

    df = gen_signal_join_df(df, sig_gen)

    plt.figure(dpi=300)
    df.plot(ax=plt.gca())
    plt.title('Traffic Variation')
    plt.savefig('Traffic_Variation_a')
