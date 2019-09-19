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
#from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

start_Time = time.time()

workPath = 'V:/2019/WIP/Edelweiss_HackerEarth_I_II/WIP/PartII'

X1 = pd.read_csv(os.path.join(workPath, "train1.csv"))
X2 = pd.read_csv(os.path.join(workPath, "train2.csv"))
X3 = pd.read_csv(os.path.join(workPath, "train3.csv"))

Dataset = pd.concat([X1,X2,X3])

Dataset.to_csv('V:/2019/WIP/Edelweiss_HackerEarth_I_II/WIP/PartII/Dataset_train.csv')

X = Dataset.iloc[:,5:]
y = Dataset.iloc[:,3]

X1_test = pd.read_csv(os.path.join(workPath, "test1.csv"))
X2_test = pd.read_csv(os.path.join(workPath, "test2.csv"))
X3_test = pd.read_csv(os.path.join(workPath, "test3.csv"))

Dataset_test = pd.concat([X1_test,X2_test,X3_test])

Dataset_test.to_csv('V:/2019/WIP/Edelweiss_HackerEarth_I_II/WIP/PartII/Dataset_test.csv')

X_test = Dataset.iloc[:,4:]