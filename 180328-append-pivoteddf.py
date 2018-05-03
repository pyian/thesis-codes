import pandas as pd
import numpy as np


def read_data(paths):
    dfs = pd.DataFrame()
    for path in paths:
        df = pd.read_hdf(path, 'table')
        dfs = dfs.append(df)
        print("READ {}".format(path))
    return dfs


if __name__ == '__main__':
    # paths = open('dataFiles.list').read().split()
    paths = ['frb_2016-10-zdcpr.h5', 'frb_2016-11-zdcpr.h5', 'frb_2016-12-zdcpr.h5']
    df = read_data(paths)
    filename = 'frb_all_zdcpr.h5'
    df.to_hdf(filename, 'table',  format='table', data_columns=True)
