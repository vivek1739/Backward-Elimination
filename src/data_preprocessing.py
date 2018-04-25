# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# getting path of raw data
rawdata_path = os.path.join(os.path.pardir,'data','raw')
processeddata_path = os.path.join(os.path.pardir,'data','processed')
dataset = pd.read_csv(os.path.join(rawdata_path,'50_Startups.csv'))

# getting X and y
X = dataset.iloc[:,:4].values
y = dataset.iloc[:,4].values

# Data has one categorical variable which needs to be removed
# ie. State
# We will label encode and one hot encode the data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder = LabelEncoder()
X[:,3] = labelEncoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# dummy variable
X = X[:,1:]

# Splitting the data randomly 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Save the processed data in data/processed
np.savetxt(os.path.join(processeddata_path,'X_train.csv'), X_train, delimiter=",")
np.savetxt(os.path.join(processeddata_path,'X_test.csv'), X_test, delimiter=",")
np.savetxt(os.path.join(processeddata_path,'y_train.csv'), y_train, delimiter=",")
np.savetxt(os.path.join(processeddata_path,'y_test.csv'), y_test, delimiter=",")

