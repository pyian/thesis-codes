import pandas as pd
import numpy as np

dtypes = {'sensor_type': 'category', 'loc_dir': 'category', 'loc_spn': 'category',
          'loc_pin': 'category', 'loc_pot': 'category', 'loc_num': 'category', 'remark': 'category'}

months = ['01']

for month in months:
    path = '/exports/tcfs01/backedup/geos_tc_bridges/frb_2016-' + month + '-zd.csv'
    print('Reading file from ...' + path)

    df = pd.read_csv(path, dtype=dtypes)
    print('File read')

    df['timestamp'] = pd.to_datetime(
        df['timestamp'], format='%Y-%m-%d %H:%M:%S')
#    df.loc_num = df.loc_num.astype('category')
    print('Data type converted')

    try:
        df = df.drop('Unnamed: 0', axis=1)
    except:
        pass

    try:
        df = df.drop('Unnamed: 0.1', axis=1)
    except:
        pass

    df.offset_value = df.offset_value.fillna(0)
    df['val'] = df.d_val - df.offset_value
    df = df.drop(['d_val', 'offset_value'], axis=1)
    print('zo applied')

    # Replace strings in col coord_x with nan
    df.coord_x = df.coord_x.replace(['TBC', 'tbc'], np.nan)
    df.coord_x = df.coord_x.astype('float')

    # Write to csv - for comparison with hdf5
#    filename = 'frb_2016-' + month + '-zdc.csv'
#    df.to_csv(filename)
#    print('Wrote to file: ' + filename)

    # Write to hdf5
    filename = 'frb_2016-' + month + '-zdc.h5'
    df.to_hdf(filename, 'table', format='table', data_columns=True)
    print('Wrote to file: ' + filename)
