# -*- coding: utf-8 -*-
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# getting path of raw data and train/test data
processeddata_path = os.path.join(os.path.pardir,'data','processed')
X_train = pd.read_csv(os.path.join(processeddata_path,'X_train.csv'),header=None).values
y_train = pd.read_csv(os.path.join(processeddata_path,'y_train.csv'),header=None).values
X_test =  pd.read_csv(os.path.join(processeddata_path,'x_test.csv'),header=None).values
y_test =  pd.read_csv(os.path.join(processeddata_path,'y_test.csv'),header=None).values

# sklearn takes the b ie. constant term automatically but in statsmodel we need to add
# a column of ones to the dataset
# SN =0.05
import statsmodels.formula.api as sm
X = np.append(X_train,X_test,axis=0)
y = np.append(y_train,y_test, axis=0)
X = np.append(arr = np.ones((X.shape[0],1)).astype(int),values=X,axis=1)
X_opt = X[:,[0,1,2,3,4,5]]
# we will use ordinary least sq regressor
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()


# x2 got highest p value
X_opt = X[:,[0,1,3,4,5]]
# we will use ordinary least sq regressor
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()

# x1
X_opt = X[:,[0,3,4,5]]
# we will use ordinary least sq regressor
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()

# now x2 shows no impact on profit
X_opt = X[:,[0,3,5]]
# we will use ordinary least sq regressor
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()

# now x2 
X_opt = X[:,[0,3]]
# we will use ordinary least sq regressor
regressor_OLS = sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()

