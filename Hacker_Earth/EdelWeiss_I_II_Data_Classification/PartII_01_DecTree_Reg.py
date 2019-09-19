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
from sklearn.metrics import r2_score,mean_absolute_error
#from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

start_Time = time.time()

step = 0
print(step)

workPath = 'V:/2019/WIP/Edelweiss_HackerEarth_I_II/WIP/PartII'

Dataset_Train = pd.read_csv(os.path.join(workPath, "Dataset_train.csv"))
Dataset_Test = pd.read_csv(os.path.join(workPath, "Dataset_test.csv"))

Dataset_Train_drop = Dataset_Train.copy()
Dataset_Test_drop = Dataset_Test.copy()

Dataset_Train_drop = Dataset_Train_drop.replace(['#NAME?','inf','Inf'], 'NaN')
Dataset_Test_drop = Dataset_Test_drop.replace(['#NAME?','inf','Inf'], 'NaN')

step += 1
print(step)

X = Dataset_Train_drop.iloc[:,6:]
y = Dataset_Train_drop.iloc[:,4]

X_test_sub = Dataset_Test_drop.iloc[:,5:]

# Taking care of missing data
from sklearn.preprocessing import Imputer
X_imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X_imputer = X_imputer.fit(X)
X = X_imputer.transform(X)

X_test_sub_imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X_test_sub_imputer = X_test_sub_imputer.fit(X_test_sub)
X_test_sub = X_test_sub_imputer.transform(X_test_sub)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

step += 1
print(step)

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train,y_train)

import pickle
address_forModel = r"V:\2019\WIP\Edelweiss_HackerEarth_I_II\WIP\PartII"
with open(address_forModel+'/ranFor_100_regModel.py', 'wb') as model:
    pickle.dump(regressor, model)

y_pred = regressor.predict(X_test) 

r2_score = r2_score(y_test, y_pred, sample_weight=None)
print(r2_score)

mean_absolute_error = mean_absolute_error(y_test, y_pred, sample_weight=None)
print(mean_absolute_error)

print("Total time" + str(time.time() - start_Time))