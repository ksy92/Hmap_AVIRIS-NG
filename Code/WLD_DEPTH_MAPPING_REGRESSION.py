
"""
Created on Tue Sep 19 13:11:44 2023

@author: Siyoon Kwon
"""
import pandas as pd
import numpy as np
import os

import pickle
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from WLD_DEPTH_MAPPING_FUNCTIONS import RFECV_PLSR, RFECV_RF, regression_results

path = 'C:\\Users\\syk32\\Siyoon_Box Dropbox\Kwon Siyoon\\WLD_DEPTH_MAPPING' #add main path
def change_dir(path):
    os.chdir(path)
    
change_dir('C:\\Users\\syk32\\Siyoon_Box Dropbox\Kwon Siyoon\\WLD_DEPTH_MAPPING')
print('dir:', os.getcwd())

#%%Data load & model selection

df_2021a = pd.read_csv(path+'\\Input\\H_Rrs_df2021a_window_3.csv', index_col=False)
df_2021b = pd.read_csv(path+'\\Input\\H_Rrs_df2021b_window_3.csv', index_col=False)

camp = '2021a'#'2021a' or '2021b' or 'total'

if camp =='2021b': #spring
    target_df = df_2021a
elif camp =='2021b': #fall
    target_df = df_2021b
elif camp =='total': #combined data
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

best_max_depth = None
best_n_estimators = None
best_rmse = float('inf')
X = spectrum.values
Y = depth.values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)


max_depth_values = list(range(1, 31, 3)) + [None]
n_estimators = 100

for max_depth in max_depth_values:
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=4,
            random_state=20
        )
        
        # Training
        estimator.fit(X_train, y_train)
        
        # Prediction
        y_train_pred = estimator.predict(X_train)
        y_test_pred = estimator.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        if test_rmse < best_rmse:
            best_max_depth = max_depth
            best_n_estimators = n_estimators
            best_rmse = test_rmse
        
        print(f'max_depth = {max_depth}, n_estimators = {n_estimators}, Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}')

print(f'Best max_depth = {best_max_depth}, Best n_estimators = {best_n_estimators}, Best Test RMSE = {best_rmse:.4f}')




#%%n_comp
X = spectrum.values
Y = depth.values
cv = 5 #cross-validation fold
step = 1 #step to test features
n_comp_list = np.arange(8)+2
# Lists to store results:
dsets = []
rfeindices = []
grid_scores_s = []

for ii in n_comp_list:
   dset, rfeindex, grid_scores_ = RFECV_PLSR(X,Y, step, cv ,ii) #def RFECV_PLSR(X,Y, step, cv, n_comp):
   dsets.append(dset)         # Save the dataset for this iteration
   rfeindices.append(rfeindex) # Save the ranking indices for this iteration
   grid_scores_s.append(grid_scores_)

np.savez(path+'\\dsets_plsr_'+camp+'.npz',*dsets)
np.savez(path+'\\rfeindices_plsr_'+camp+'.npz', *rfeindices)
np.savez(path+'\\grid_scores_s_plsr_'+camp+'.npz', *grid_scores_s)

#%%
data = np.load(path+'\\dsets_plsr.npz', allow_pickle=True)
dsets_plsr = [data[f'arr_{i}'] for i in range(8)]

# data = np.load(path+'\\rfeindices_plsr.npz', allow_pickle=True)
# rfeindices_plsr = [data[f'arr_{i}'] for i in range(8)]

data = np.load(path+'\\grid_scores_s_plsr.npz', allow_pickle=True)
grid_scores_s_plsr = [data[f'arr_{i}'] for i in range(8)]

#%%
# frame = np.zeros(len(grid_scores_s_plsr), 90, )
grid_scores_s_plsr2 = grid_scores_s_plsr.copy()
dummy = np.zeros_like(grid_scores_s_plsr[0])+np.nan

for i in np.arange(len(grid_scores_s_plsr)):
   if len(grid_scores_s_plsr[i]) < len(grid_scores_s_plsr[0]):
      gap = len(grid_scores_s_plsr[0]) - len(grid_scores_s_plsr[i])
      
      
      # dummy[gap:] = grid_scores_s_plsr[i]
      grid_scores_s_plsr2[i] = np.concatenate([np.full(gap, np.nan), grid_scores_s_plsr[i]])
      

grid_scores_s_plsr3 = np.reshape(np.concatenate(grid_scores_s_plsr2), (len(grid_scores_s_plsr), 90))
#%%Plot grid
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 14
plt.figure(figsize=(8, 3)) 
plt.pcolormesh(np.sqrt(-grid_scores_s_plsr3), cmap='jet_r', vmin=3.5, vmax=5.5)  # 'viridis' is a colormap, you can choose any other vmin=4.5, vmax=6
plt.colorbar()  

plt.yticks(np.arange(0.5, len(grid_scores_s_plsr3), 1), [f'{int(val)}' for val in np.linspace(2, 9, len(grid_scores_s_plsr3))])

plt.show()

grid_scores_s_plsr4 = np.sqrt(-grid_scores_s_plsr3)
min_value = np.nanmin(grid_scores_s_plsr4)
min_position = np.unravel_index(np.nanargmin(grid_scores_s_plsr4), grid_scores_s_plsr4.shape)  # 최대값의 위치 찾기, nan은 무시

# 결과 출력
print("Min value:", min_value)
print("(raw, col):", min_position)

#%%
rfeindex = pd.read_csv(path+'\\Model\\rfeindex_plsr.csv')


total = np.floor(rfeindex['Wave'][rfeindex['Total'] == 1])
wave_2021a = np.floor(rfeindex['Wave'][rfeindex['Spring'] == 1])
wave_2021b = np.floor(rfeindex['Wave'][rfeindex['Fall'] == 1])
#%%Model selection
model_opt = '2' #1: PLSR, 2: RF
best_max_depth = 20

if model_opt == '1':  
    estimator = PLSRegression(n_components=8)#
    model = 'PLSR'
elif model_opt == '2':  
    estimator = RandomForestRegressor(n_estimators = 50, max_depth=best_max_depth, n_jobs=-1, random_state=20) # you can customize all hyperparameters.
    model = 'RF'
    
#%%Feature Selction - RFE-CV
X = spectrum.values
Y = depth.values
cv = 5 #cross-validation fold
step = 1 #step to test features

if model_opt == '1':  
    dset, rfeindex = RFECV_PLSR(X,Y, step, cv, estimator, 8)
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



