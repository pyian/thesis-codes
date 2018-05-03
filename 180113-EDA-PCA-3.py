# Calcaulate PCA

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt

filename = '/exports/tcfs01/backedup/geos_tc_bridges/frb_2016-02-zdc.h5'

col = ['timestamp', 'sensor_id', 'val']

df = pd.read_hdf(filename, 'table', mode='r', columns=col)

dates = pd.DatetimeIndex(df.timestamp)
month = dates.month.unique()[0]

print(df.info())

df_piv = pd.pivot_table(df, values='val', index=[
                        'timestamp'], columns=['sensor_id'])
df_piv_resampled = df_piv.resample('10S').mean()
df_piv_resampled = df_piv_resampled.dropna()

X = df_piv_resampled.values
pca = PCA(n_components=20)

Y = pca.fit(X)
var = pca.explained_variance_ratio_
var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

plt.plot(var1)
plt.xlabel('n_components')
plt.ylabel('Variance Ratio')
title = ('Month-' + str(month) + ' Variance Ratio vs Number of Components')
plt.title(title)
plt.savefig('figs/fig-' + title.replace(' ', '_') + '.png')
print('Figures saved.')

# Coorelation
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

print('Script Finished.')
