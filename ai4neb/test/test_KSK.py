#%%
import numpy as np
import pandas as pd

from ai4neb import manage_RM, manage_data
#%%
def get_sets(func='sins', n_samples=200, verbose=True):
    n_samples=200
    if func == 'sins':
        X1 = np.random.randn(n_samples)
        y1 = np.sin(X1) + np.cos(X1) + 0.1*np.random.randn(n_samples)
        X2 = np.random.randn(n_samples, 2)
        y2 = np.array([np.sin(X2[:,0]) + np.cos(X2[:,1]) + 0.1*np.random.randn(n_samples),
             np.sin(X2[:,0])**2 + np.cos(X2[:,1])**2 + 0.1*np.random.randn(n_samples)]).T
    elif func == 'x24':
        y1 = np.random.uniform(-5, 5, n_samples)
        X1 = y1**2
        y2 = np.random.uniform(-5, 5, (n_samples, 2))
        X2 = np.array([y1**2, y1**4]).T
        
    if verbose:
        print('X1 shape:', X1.shape)
        print('X2 shape:', X2.shape)
        print('y1 shape:', y1.shape)
        print('y2 shape:', y2.shape)
    return X1, y1, X2, y2
#%%
X1, y1, X2, y2 = get_sets()
# %%
RM = manage_RM('KSK_ANN', X2, y1)
# %%
RM.init_RM()
# %%
RM.train_RM()
# %%
