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

#loading csvfile (replace first parameter with actual file name)
sydney_df = pd.read_csv(csvfile, 
                        names = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16',
                                 'Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Y10','Y11','Y12','Y13','Y14','Y15','Y16',
                                 'P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16',
                                 'P_total'])

sydney_df = sydney_df.astype(float)

# define input and output variables
X = sydney_df[['X1','X2','X3','X4','X5','X6','X7','X8',
               'X9','X10','X11','X12','X13','X14','X15','X16',
               'Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8',
               'Y9','Y10','Y11','Y12','Y13','Y14','Y15','Y16']]
Y = sydney_df[['P1','P2','P3','P4','P5','P6','P7','P8',
               'P9','P10','P11','P12','P13','P14','P15','P16','P_total']]

# split dataset into training and validation
## 80% for training and remaining 20% for validation
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.20)

# create model
regression_model = LinearRegression()
regression_model.fit(X_train,Y_train)

# get predictions
y_predict = regression_model.predict(X_test)

# visualizing the coefficient matrix
fig, ax = plt.subplots(figsize = (20,10))
xlabels = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P_total']
ylabels = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16',
           'Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Y10','Y11','Y12','Y13','Y14','Y15','Y16']
sns.heatmap(np.transpose(regression_model.coef_), cmap="RdBu", xticklabels = xlabels, yticklabels = ylabels, linewidths = 0.5)

# np.matmul(regression_model.coef_,X_test[:1])
print(mean_absolute_percentage_error(y_predict, y_test))
     
observed_data = np.concatenate((X_test, Y_test), axis = 1)
predicted_data = np.concatenate((X_test, train.predict(X_test)), axis = 1)

observed_df = pd.DataFrame(observed_data,
                           columns = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16',
                                      'Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9', 'Y10','Y11','Y12', 'Y13','Y14','Y15','Y16',
                                      'P1','P2','P3', 'P4','P5','P6','P7','P8', 'P9','P10','P11','P12','P13','P14','P15','P16','P_total'])
predicted_df = pd.DataFrame(predicted_data, 
                            columns = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16',
                                       'Y1','Y2','Y3','Y4''Y5','Y6','Y7','Y8','Y9','Y10','Y11','Y12','Y13','Y14','Y15','Y16', 
                                       'P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P_total'])

# plot of P_total
t = np.linspace(1325000,1500000, num = 2)
plt.scatter(predicted_df['P_total'], observed_df['P_total'], alpha = 0.3, s = 0.5)
plt.plot(t,t,'r--')
plt.show()

# plots of P1 - P16
# the model is way better at predicting individual power outputs than the total (if you look at the coefficient matrix, it is assigning drastically different weights to the x and y coordinates even though, in theory, each WEC contributes equally to P_total)
t = np.linspace(50000,120000, num = 2)
for i in range (1,17):
  plt.subplot(4,4,i)
  plt.scatter(predicted_df['P'+str(i)], observed_df['P'+str(i)], alpha = 0.3, s = 0.5)
  plt.plot(t,t,'r--')
plt.show()
