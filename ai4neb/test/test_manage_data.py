import numpy as np
import pandas as pd



from ai4neb import manage_RM, manage_data
#%%

X = pd.DataFrame([[1,2,3], [4,5,6], [7,8,9]])
Xcn = pd.DataFrame([[1,2,3], [4,5,6], [7,8,9]], columns = ['X1', 'X2', 'y'])
y = pd.DataFrame([1,2,3],)


dd = manage_data(X=X, y=y)
dd2 = manage_data(dataframe=X, X_str=[0, 1], y_str=[2,])
dd3 = manage_data(dataframe=X, X_str=['X1', 'X2'], y_str=['y',])



# %%
