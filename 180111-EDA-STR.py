# Plot STR

from sys import argv
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
# import numpy as np

filename = argv[1]  # 0 is script name, 1 is param name

col = ['timestamp', 'sensor_id', 'loc_dir',
       'loc_pin', 'loc_spn', 'loc_pot', 'val']
# CHANGE THIS
df = pd.read_hdf(filename, 'table',
                 where='sensor_type in str', mode='r', columns=col)

print(df.info())

# Extract month
dates = pd.DatetimeIndex(df.timestamp)
month = dates.month.unique()[0]


# Describe df
df_piv = pd.pivot_table(df, values='val',
                        index=['timestamp'], columns=['sensor_id'])
print(df_piv.describe())

# plot
directions = ['NW', 'NE', 'SW', 'SE']
spans = ['MS', 'SS']
pots = ['inner', 'outer']
pins = ['bot', 'top']

for pin in pins:
    for pot in pots:
        for span in spans:
            for direction in directions:
                df_str = (df[(df['loc_dir'] == direction)
                             & (df['loc_spn'] == span)
                             & (df['loc_pot'] == pot)
                             & (df['loc_pin'] == pin)])

                fig = plt.figure(dpi=300)
                ax = fig.add_subplot(111)

                for sensor in df_str.sensor_id.unique():
                    x = df_str[df_str['sensor_id'] == sensor].timestamp
                    y = df_str[df_str['sensor_id'] == sensor].val

                    ax.plot_date(x, y, alpha=0.5, linewidth=0.5, label=sensor,
                                 linestyle='-', marker="")
                    ax.legend()

                title = ('Month-' + str(month) + '-Strain variation in '
                         + direction + ' ' + span + ' at ' + pot + ' post on '
                         + pin + ' pin')
                plt.title(title)
                plt.xlabel('Time')
                plt.ylabel('Strain (ms)')
#                 plt.ylim(-20, 40)
                fig.savefig('figs/fig-' + title.replace(' ', '_') + '.png')
print('Figures saved.')
# corr

directions = ['NW', 'NE', 'SW', 'SE']
spans = ['MS', 'SS']
pots = ['inner', 'outer']
pins = ['bot', 'top']

for pin in pins:
    for pot in pots:
        for span in spans:
            for direction in directions:
                df_corr = (df[(df['loc_dir'] == direction)
                              & (df['loc_spn'] == span)
                              & (df['loc_pot'] == pot)
                              & (df['loc_pin'] == pin)])
                df_corr = pd.pivot_table(df_corr, values='val', index=[
                                         'timestamp'], columns=['sensor_id'])
                print('Sensor Group Correlation: ' + direction + ' ' +
                      span + ' at ' + pot + ' post on ' + pin + ' pin')
                print(df_corr.corr())
print('Script Finished.')
