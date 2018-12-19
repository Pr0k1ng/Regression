# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 10:11:54 2018

@author: abhis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding the independent variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# avoiding dummy variable trap

X= X[:,1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)
