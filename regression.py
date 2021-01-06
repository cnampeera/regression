#importing libraries

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pylab
import math

from scipy import stats
from scipy.optimize import minimize
# PyMC3 for Bayesian Inference
import pymc3 as pm

import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

sydney_df = sydney_df.astype(float)

# define input and output variables
X = sydney_df[['X1','X2','X3','X4','X5','X6','X7','X8',
               'X9','X10','X11','X12','X13','X14','X15','X16',
               'Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8',
               'Y9','Y10','Y11','Y12','Y13','Y14','Y15','Y16']]
y = sydney_df[['P1','P2','P3','P4','P5','P6','P7','P8',
               'P9','P10','P11','P12','P13','P14','P15','P16','P_total']]

# split dataset into training and validation
## 80% for training and remaining 20% for validation
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20)

# create model
regression_model = LinearRegression()
regression_model.fit(X_train,y_train)

# get predictions
y_predict = regression_model.predict(X_test)

# this is kind of good for visualizing the coefficient matrix instead of printing it out?
fig, ax = plt.subplots(figsize = (20,10))
xlabels = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P_total']
ylabels = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16',
           'Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Y10','Y11','Y12','Y13','Y14','Y15','Y16']
sns.heatmap(np.transpose(regression_model.coef_), cmap="RdBu", xticklabels = xlabels, yticklabels = ylabels, linewidths = 0.5)

# defining MAPE
def mean_absolute_percentage_error(y_pred, y_true, sample_weights = None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)
    
    if np.any(y_true==0):
        print("Found zeroes in y_true. MAPE undefined. Removing from set...")
        idx = np.where(y_true==0)
        y_true = np.delete(y_true, idx)
        y_pred = np.delete(y_pred, idx)
        if type(sample_weights) != type(None):
            sample_weights = np.array(sample_weights)
            sample_weights = np.delete(sample_weights, idx)
        
    if type(sample_weights) == type(None):
        return(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    else:
        sample_weights = np.array(sample_weights)
        assert len(sample_weights) == len(y_true)
        return(100/sum(sample_weights)*np.dot(
                sample_weights, (np.abs((y_true - y_pred) / y_true))
        ))
