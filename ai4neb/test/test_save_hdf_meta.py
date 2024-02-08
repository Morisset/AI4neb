#%%
import pytest

import pandas as pd
import numpy as np
#%%

df = pd.DataFrame(np.arange(10), columns=['data1'])
df.attrs.update({'test1':0,'test2':'this is a string'})
#%%
# Save file
def test_save_h5(df):
   with pd.HDFStore('test.h5') as hdf_store:
      hdf_store.put('data', df, format='table')
      hdf_store.get_storer('data').attrs.metadata = df.attrs
#%%
# Load file and meta
def test_read_h5():
   with pd.HDFStore('test.h5') as hdf_store:
      metadata = hdf_store.get_storer('data').attrs.metadata
      df_read = hdf_store.get('data')
      # If you want to set up meta in the df again
      df_read.attrs.update(metadata)
      return df_read
# %%
# %%
# %%
