
"""
Created on Tue Sep 19 13:11:44 2023

@author: Siyoon Kwon
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('C:\\Users\\syk32\\Dropbox\\Delta-X\\Code')

# import osgeo
from osgeo import gdal, osr
# import spectral
import spectral.io.envi as envi 
# from spectral import*
import scipy.spatial.transform._rotation_groups
import scipy
import scipy.special.cython_special
import glob
import os
import matplotlib.pyplot as plt

#%%CMR-OV
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
# from xgboost.sklearn import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
# import xgboost as xgb
import pickle
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import check_cv
from sklearn.base import clone, is_classifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score



from WLD_DEPTH_MAPPING_FUNCTIONS import RFECV_PLSR, RFECV_RF, regression_results

#%%Data load & model selection

df_2021a = pd.read_csv('C:\\Users\\syk32\\Dropbox\\Delta-X\\ML_MODEL\\H_Rrs_df2021a_window_3.csv', index_col=False)
df_2021b = pd.read_csv('C:\\Users\\syk32\\Dropbox\\Delta-X\\ML_MODEL\\H_Rrs_df2021b_window_3.csv', index_col=False)

camp = 'total'#'2021a' or '2021b' or 'total'

if camp =='2021a':
    target_df = df_2021a
elif camp =='2021b':
# total_df.iloc[:,0] = 1
    target_df = df_2021b
elif camp =='total':
# total_df2.iloc[:,0] = 2
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
    
    # 모델 학습
    estimator.fit(X_train, y_train)
    
    # 훈련 세트와 테스트 세트에 대한 예측
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
model_opt = '1' #1: PLSR, 2: RF
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
    
np.save('H_rfeindex_'+camp+'_'+model+'_win3.npy', rfeindex)

#%%Train
test_size = 0.2

rfeindex = np.load('H_rfeindex_'+camp+'_'+model+'_win3.npy')#np.load('C:\\Users\\syk32\\Dropbox\\Delta-X\\ML_MODEL\\H_rfeindex_deriv_total_win5.npy')# #이전버젼: tss_rfeindex_deriv_2 #'C:\\Users\\syk32\\Dropbox\\Delta-X\\ML_MODEL\\H_rfeindex_deriv_total_win3.npy'
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
filename = 'H_estimator_DELTA_X_' + camp + '_win3.sav'
pickle.dump(estimator, open(filename, 'wb'))
#%%geopandas df with prediction
from shapely.geometry import Point
import geopandas as gpd
from shapely.wkt import loads

y_rs_win3 = estimator.predict(spectrum.values)

target_df['win3_pred'] = y_rs_win3

target_df['geometry'] = target_df['geometry'].apply(lambda x: loads(x))

target_gdf = gpd.GeoDataFrame(target_df, geometry='geometry')
target_gdf.to_file('H_regression_result'+camp+'_'+model+'_win3.shp')
    
#%%Cross-validation!
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import pickle

filename = 'C:\\Users\\syk32\\Dropbox\\Delta-X\\ML_MODEL\\H_RF_DELTA_X_2021b.sav'
loaded_model = pickle.load(open(filename, 'rb'))
rfeindex_0 = np.arange(91)#np.load('C:\\Users\\syk32\\Dropbox\\Delta-X\\ML_MODEL\\H_rfeindex_deriv_2021b.npy')

for z in np.array([0]):#[n_clusters]
    X1 = X.copy()#spectrum[case_for_cl['Cluster']==z] #+1
    X = pd.DataFrame(X1)#.iloc[:,rfeindex_0]
    Y = con#[case_for_cl['Cluster']==z]
    globals()['estimator_{}'.format(z)] =  model1
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=20)
    globals()['CV_score_{}'.format(z)] = cross_val_score(globals()['estimator_{}'.format(z)], X, Y, cv=cv)

#%%
from scipy.stats import gaussian_kde
import matplotlib
import matplotlib.lines as mlines

# Generate fake data
# y1=y_test
# y2=y_pred

y1=y_train
y2=y_pred2
x = np.array(y1).reshape(-1)
y = np.array(y2).reshape(-1)

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots(figsize=(7, 6))
plt.style.use('default')
ax.axis('on')

ax.set_facecolor('w')
font = {'fontname':'Ariel'}
# font2 = font_manager.FontProperties(family='Times New Roman',
#                                     size=15)
plt.rcParams['font.family']='Ariel'
# fig, ax = plt.subplots(figsize=(10, 10))
line = mlines.Line2D([0, 1], [0, 1], color='black', linestyle='--', linewidth=0.5)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.scatter(x, y, c=z, s=20, cmap=plt.cm.jet)
# ax.set_yscale('log')
# ax.set_xscale('log')


plt.xticks(fontname = "Ariel", size=15)
plt.yticks(fontname = "Ariel", size=15)
plt.ylim(0,30);
plt.xlim(0,30);
ax.set_ylabel('Estimated depth (m)', fontsize = 20,fontdict=font)
ax.set_xlabel('Measured depth (m)', fontsize = 20,fontdict=font)
# ax.figure(figsize = (8, 8))
plt.show()

