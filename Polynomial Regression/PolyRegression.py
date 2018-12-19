# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 20:00:58 2018

@author: abhis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"from sklearn.cross_validation import train_test_split
"X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#Linear 

#Polynomial

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)