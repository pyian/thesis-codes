# Pivot the data by sensor_id

import pandas as pd
import numpy as np
from sys import argv

"""
Seperate df into chucks
Pivot table
by sensor_id
resample to 30S
append dfs
resample again
save df
"""


def clean_weather(df):
    # clean weather id (has duplicates)
    def set_null(x):
        if x == 272:
            return 0
        else:
            return x

    def replace_id(x):
        if x == 1604:
            return 9001
        elif x == 1606:
            return 9002
        elif x == 1625:
            return 9011
        elif x == 1626:
            return 9012
        else:
            return 0

    df['new_sen_id'] = df['sensor_id'].apply(set_null)
    df['weather_id'] = df['d_st'].apply(replace_id)
    df['new'] = df['new_sen_id'] + df['weather_id']
    df['id'] = df['sensor_type'].astype('str') + '_' + df['new'].astype('str')
    return df


def pivot_resample(df):
    df = df.pivot_table(values='val', index='timestamp', columns='id')
    df = df.resample('30S').mean()
    print("Pivoted and Resampled")
    return df


def main(df_iter):

    df = pd.DataFrame()

    for iter_num, chuck in enumerate(df_iter, 1):
        print("Processing iteration: {}".format(iter_num))
        
        chuck = clean_weather(chuck)
        chuck = chuck.drop(['sensor_id', 'd_st', 'new_sen_id',
                        'weather_id', 'new', 'sensor_type'], axis=1)
        chuck = pivot_resample(chuck)
        df = df.append(chuck)
        del [chuck]

    # resample again
    df = df.resample('30S').mean()

    return df




if __name__ == '__main__':
    
    # Read path
    # path = 'frb_2016-02-zdc.h5'
    path = argv[1]
    path = "/exports/tcfs01/backedup/geos_tc_bridges/" + path

    # Read file
    col = ['timestamp', 'sensor_id', 'val', 'sensor_id', 'd_st', 'sensor_type']
    df_iter = pd.read_hdf(path, 'table', columns=col, iterator=True)
    print('Read from {}'.format(path))

    # Process
    df = main(df_iter)

    # Save data
    filename = path[:-3] + 'pr.h5'
    df.to_hdf(filename, 'table',  format='table', data_columns=True)
    print('File wrote to {}\nEnd of Script.'.format(filename))
