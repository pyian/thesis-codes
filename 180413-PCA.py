"""
Do PCA on everything
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

df = pd.read_hdf('frb_all_zdcprdt.h5', 'table')

def do_pca(X):
    pca = PCA(n_components=8).fit(X)
    print(pca.explained_variance_ratio_)

do_pca(df.dropna().values)

print('End of Script.')
