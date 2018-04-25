# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# getting path of raw data and train/test data
processeddata_path = os.path.join(os.path.pardir,'data','processed')
X_train = pd.read_csv(os.path.join(processeddata_path,'X_train.csv')).values
y_train = pd.read_csv(os.path.join(processeddata_path,'y_train.csv')).values
X_test =  pd.read_csv(os.path.join(processeddata_path,'x_test.csv')).values
y_test =  pd.read_csv(os.path.join(processeddata_path,'y_test.csv')).values

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# getting r2 score for regression
from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)
