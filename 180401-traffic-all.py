"""
Combine all traffic data
"""

import pandas as pd
import numpy as np


def read_data(paths):
    dfs = pd.DataFrame()
    for path in paths:
        df = pd.read_csv(path)
        # to datetime
        df.timestamp = pd.to_datetime(df.timestamp)
        df = df.set_index('timestamp')
        dfs = dfs.append(df)
        print("READ {}".format(path))
    return dfs


if __name__ == '__main__':
    paths = ["tfc-r-2016-10.csv", "tfc-r-2016-9.csv", "tfc-r-2017-3.csv",
             "tfc-r-2017-6.csv", "tfc-r-2016-11.csv", "tfc-r-2017-1.csv",
             "tfc-r-2017-4.csv", "tfc-r-2017-7.csv", "tfc-r-2016-12.csv",
             "tfc-r-2017-2.csv", "tfc-r-2017-5.csv", "tfc-r-2017-8.csv"]
    df = read_data(paths)
    filename = 'tfc-r-all.h5'
    df = df.sort_index()
    df.to_hdf(filename, 'table',  format='table', data_columns=True)
