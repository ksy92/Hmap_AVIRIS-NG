
"""
Created on Tue Sep 19 13:11:44 2023

@author: Siyoon Kwon
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score

from WLD_DEPTH_MAPPING_FUNCTIONS import RFECV_PLSR, RFECV_RF, regression_results

path = 'C:\\Users\\syk32\\Dropbox\\\WLD_DEPTH_MAPPING'
def change_dir(path):
    os.chdir(path)
    
change_dir('C:\\Users\\syk32\\Dropbox\\\WLD_DEPTH_MAPPING')
print('dir:', os.getcwd())

#%%Data load & model selection

df_2021a = pd.read_csv(path+'\\Input\\H_Rrs_df2021a_window_3.csv', index_col=False)
df_2021b = pd.read_csv(path+'\\Input\\H_Rrs_df2021b_window_3.csv', index_col=False)

camp = 'total'#'2021a' or '2021b' or 'total'

if camp =='2021a':
    target_df = df_2021a
elif camp =='2021b':
    target_df = df_2021b
elif camp =='total':
    target_df = pd.concat([df_2021a, df_2021b])

target_df =target_df[target_df['river_dept']>0]

depth = target_df['river_dept']
spectrum = target_df.iloc[:,-91:]

# Filter rows with NaN values
nan_mask_x = np.isnan(spectrum).any(axis=1)
nan_mask_y = np.isnan(depth)
depth = depth[~nan_mask_x]
spectrum = spectrum[~nan_mask_x]

#%%hyperparameter tuning for RF
from sklearn.metrics import mean_squared_error

best_max_depth = None
best_rmse = float('inf')
X = spectrum.values
Y = depth.values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,  random_state=20)

for max_depth in range(1, 31):
    estimator = RandomForestRegressor(
        n_estimators=50,   
        max_depth=max_depth,  
        n_jobs=4,            
        random_state=20      
    )
    
    #Training
    estimator.fit(X_train, y_train)
    
    # Prediction
    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    if test_rmse < best_rmse:
        best_max_depth = max_depth
        best_rmse = test_rmse
    
    print(f'max_depth = {max_depth}, Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}')

print(f'Best max_depth = {best_max_depth}, Best Test RMSE = {best_rmse:.4f}')

#%%Model selection
model_opt = '2' #1: PLSR, 2: RF
best_max_depth = 20

if model_opt == '1':  
    estimator = PLSRegression(n_components=91)#
    model = 'PLSR'
elif model_opt == '2':  
    estimator = RandomForestRegressor(n_estimators = 50,max_depth=best_max_depth, n_jobs=4, random_state=20) # you can customize all hyperparameters.
    model = 'RF'
    
#%%Feature Selction - RFE-CV
X = spectrum.values
Y = depth.values
cv = 5 #cross-validation fold
step = 1 #step to test features

if model_opt == '1':  
    dset, rfeindex = RFECV_PLSR(X,Y, step, cv, estimator)
    model = 'PLSR'
elif model_opt == '2':  
    dset, rfeindex = RFECV_RF(X,Y, step, cv, estimator)
    model = 'RF'
    
# np.save('H_rfeindex_'+camp+'_'+model+'_win3.npy', rfeindex)

#%%Train
test_size = 0.2

rfeindex = np.load(path+'\\Model\\H_rfeindex_'+camp+'_'+model+'_win3.npy')#np.load('C:\\Users\\syk32\\Dropbox\\Delta-X\\ML_MODEL\\H_rfeindex_deriv_total_win5.npy')# #이전버젼: tss_rfeindex_deriv_2 #'C:\\Users\\syk32\\Dropbox\\Delta-X\\ML_MODEL\\H_rfeindex_deriv_total_win3.npy'
X = spectrum.iloc[:,rfeindex]
Y = depth.values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,  random_state=20)

estimator.fit(X_train, y_train)
y_pred = pd.DataFrame(estimator.predict(X_test))
y_train_pred = pd.DataFrame(estimator.predict(X_train))

res_test = regression_results(y_test, y_pred)
res_train = regression_results(y_train, y_train_pred)

print(res_train)
print(res_test)

#save the model to disk
# filename = path+'\\Model\\H_'+model+'_DELTA_X_' + camp + '_win3.sav'
# pickle.dump(estimator, open(filename, 'wb'))

    
#%%Cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import pickle

filename = path+'\\Model\\H_'+model+'_DELTA_X_' + camp + '_win3.sav'
loaded_model = pickle.load(open(filename, 'rb'))
rfeindex_0 = np.load(path+'\\Model\\H_rfeindex_'+camp+'_'+model+'_win3.npy')

X = spectrum.values
Y = depth.values
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=20)
CV_score = cross_val_score(estimator, X, Y, cv=cv)

print('CV score: {:.3f} ± {:.3f}'.format(np.mean(CV_score), np.std(CV_score)))



