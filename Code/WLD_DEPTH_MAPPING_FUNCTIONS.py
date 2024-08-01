# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:50:03 2024

@author: Siyoon Kwon
"""

import pandas as pd
import numpy as np
# from spectral import*
import scipy.spatial.transform._rotation_groups

import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import check_cv
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.base import clone, is_classifier


def regression_results(y_true, y_pred):
    errors = abs(np.array(y_true).reshape(-1) - np.array(y_pred).reshape(-1))
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = np.mean(errors)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    mape = 100 * (errors / np.array(y_true).reshape(-1))
    mape = np.mean(mape)
    mean_bias_error = np.mean(np.array(y_pred).reshape(-1) - np.array(y_true).reshape(-1))
    
    return round(explained_variance, 4), round(r2, 4), round(mean_absolute_error, 4), round(np.sqrt(mse), 4), round(mape, 4), round(mean_bias_error, 4)


def RFECV_RF(X,Y, step, cv, estimator, n_jobs):
    rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring="neg_mean_squared_error", n_jobs=n_jobs)
    rfecv.fit(X, Y)

    dset = pd.DataFrame()
    dset['attr'] = np.arange(len(X[0,:]))
    dset['selection'] = rfecv.support_
    dset['ranking'] = rfecv.ranking_
    dset['importance'] = estimator.feature_importances_

    X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
    rfeindex = np.where(rfecv.support_ == True)[0]
    return dset, rfeindex

def RFECV_PLSR(X,Y, step, cv, n_comp):

    scoring="neg_mean_squared_error"
    # comp_range = np.arange(8)+2
    # n_features_to_select = 91
    estimator = PLSRegression(n_components=n_comp)
    cv = check_cv(cv, Y, classifier=is_classifier(estimator))
    scores = []
    n_features_list = []
    # comp_scores = []
    # comp_featuers = []
    # # Generate the different feature subsets and train/evaluate models
    # for comp in comp_range:
    for n_features_to_select in range(n_comp, X.shape[1] + 1, step):
            rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
            rfe.fit(X, Y)
            # Cross-validate the model and store the average score
            cv_scores = cross_val_score(rfe.estimator_, X[:, rfe.support_], Y, cv=cv, scoring=scoring)
            scores.append(np.mean(cv_scores))
            n_features_list.append(n_features_to_select)
            
    # Find the optimal number of features based on cross-validation scores
    n_features_ = n_features_list[np.argmax(scores)]
    grid_scores_ = scores
    # Refit the model with the optimal number of features
    estimator_ = PLSRegression(n_components=n_comp)#clone(estimator)
    rfe = RFE(estimator=estimator_, n_features_to_select=n_features_, step=step)
    rfe.fit(X, Y)
    support_ = rfe.support_
    ranking_ = rfe.ranking_
    rfeindex = np.where(support_ == True)[0]
    
    dset = pd.DataFrame()
    
    dset['attr'] = np.arange(len(X[0,:]))
    dset['selection'] = rfe.support_

    # dset['importance'] = estimator.score(X, Y)
    return dset, rfeindex, grid_scores_