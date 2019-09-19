# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:53:23 2019

@author: kzt9qh
"""

import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *
import pickle
#from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

start_Time = time.time()

address_SET_1 = r"V:\2019\WIP\Edelweiss_HackerEarth_I_II\WIP\PartII"
with open(address_SET_1+'/ranFor_100_regModel.py', 'rb') as f:
    regressor = pickle.load(f)

step = 0
print(step)

Dataset_test = pd.read_csv(os.path.join(address_SET_1, "Dataset_test.csv"))

Dataset_test_drop = Dataset_test.copy()

Dataset_test_drop = Dataset_test_drop.replace(['#NAME?','inf','Inf'], 'NaN')

step += 1
print(step)

X = Dataset_test_drop.iloc[:,5:]

# Taking care of missing data
from sklearn.preprocessing import Imputer
X_imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X_imputer = X_imputer.fit(X)
X = X_imputer.transform(X)

step += 1
print(step)

y = regressor.predict(X) 

print("Total time" + str(time.time() - start_Time))