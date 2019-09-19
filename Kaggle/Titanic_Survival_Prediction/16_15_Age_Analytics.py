# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:17:16 2019

@author: kzt9qh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
#from keras.models import Sequential
#from keras.layers import Dense
from sklearn.metrics import accuracy_score
#from xgboost import XGBClassifier

# Importing the dataset
dataset = pd.read_csv('Age_Reg_Train.csv')
dataset_submission = pd.read_csv('Age_Reg_Test.csv')

# Extracting the Mr Mrs details for training set

datset_NamingAddress = pd.DataFrame.copy(dataset)
commaaddress = dotaddress = 0
for individuals in range(0,len(datset_NamingAddress)):
    individualName = datset_NamingAddress.iloc[individuals,[2]].values[0]
    for findingAddress in range(0,len(individualName)):
        if str(individualName[findingAddress]) == ',':
            commaaddress = findingAddress
        elif str(individualName[findingAddress]) == '.':
            dotaddress = findingAddress
            continue
    title = individualName[commaaddress+2:dotaddress]
    
    if individualName[commaaddress+2:dotaddress] in ['Mr', 'Mrs', 'Miss', 'Master']:
        datset_NamingAddress.iloc[[individuals],[2]] = individualName[commaaddress+2:dotaddress]
    elif individualName[commaaddress+2:dotaddress] in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir']:
        datset_NamingAddress.iloc[[individuals],[2]] = 'Mr'
    elif individualName[commaaddress+2:dotaddress] in ['the Countess', 'Mme', 'Mrs. Martin (Elizabeth L', 'Lady', 'Dona']:
        datset_NamingAddress.iloc[[individuals],[2]] = 'Mrs'
    elif individualName[commaaddress+2:dotaddress] in ['Mlle', 'Ms']:
        datset_NamingAddress.iloc[[individuals],[2]] = 'Miss'
    elif str(individualName[commaaddress+2:dotaddress]) =='Dr':
        if datset_NamingAddress.iloc[[individuals],[3]].values =='Male':
            datset_NamingAddress.iloc[[individuals],[2]] = 'Mr'
        else:
            datset_NamingAddress.iloc[[individuals],[2]] = 'Mrs'
            
# Extracting the Mr Mrs details for test set
datset_submission_NamingAddress = pd.DataFrame.copy(dataset_submission)
commaaddress_s = dotaddress_s = 0
for individuals_s in range(0,len(datset_submission_NamingAddress)):
    individualName_submission = dataset_submission.iloc[individuals_s,[2]].values[0]
    for findingAddress_submission in range(0,len(individualName_submission)):
        if str(individualName_submission[findingAddress_submission]) == ',':
            commaaddress_s = findingAddress_submission
        elif str(individualName_submission[findingAddress_submission]) == '.':
            dotaddress_s = findingAddress_submission
            continue
    if individualName_submission[commaaddress_s+2:dotaddress_s] in ['Mr', 'Mrs', 'Miss', 'Master']:
        datset_submission_NamingAddress.iloc[[individuals_s],[2]] = individualName_submission[commaaddress_s+2:dotaddress_s]
    elif individualName_submission[commaaddress_s+2:dotaddress_s] in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        datset_submission_NamingAddress.iloc[[individuals_s],[2]] = 'Mr'
    elif individualName_submission[commaaddress_s+2:dotaddress_s] in ['the Countess', 'Mme', 'Mrs. Martin (Elizabeth L', 'Dona']:
        datset_submission_NamingAddress.iloc[[individuals_s],[2]] = 'Mrs'
    elif individualName_submission[commaaddress_s+2:dotaddress_s] in ['Mlle', 'Ms']:
        datset_submission_NamingAddress.iloc[[individuals_s],[2]] = 'Miss'
    elif str(individualName_submission[commaaddress_s+2:dotaddress_s]) =='Dr':
        if datset_submission_NamingAddress.iloc[[individuals_s],[3]].values =='Male':
            datset_submission_NamingAddress.iloc[[individuals_s],[2]] = 'Mr'
        else:
            datset_submission_NamingAddress.iloc[[individuals_s],[2]] = 'Mrs'
            
        
# Adding the fare Details for training set in blanks
            
for fare in range(0,len(datset_NamingAddress)):
    if str(datset_NamingAddress.iloc[fare,[7]].values[0]) == 'nan':
        datset_NamingAddress.iloc[fare,[7]] = 7.8875
    datset_NamingAddress.iloc[fare,[7]] = ((datset_NamingAddress.iloc[fare,[7]].values[0])/(datset_NamingAddress.iloc[fare,[5]].values[0]+datset_NamingAddress.iloc[fare,[6]].values[0]+1))

# Adding the fare Details for test set in blanks
            
for fare in range(0,len(datset_submission_NamingAddress)):
    if str(datset_submission_NamingAddress.iloc[fare,[7]].values[0]) == 'nan':
        datset_submission_NamingAddress.iloc[fare,[7]] = 7.8875
    datset_submission_NamingAddress.iloc[fare,[7]] = ((datset_submission_NamingAddress.iloc[fare,[7]].values[0])/(datset_submission_NamingAddress.iloc[fare,[5]].values[0]+datset_submission_NamingAddress.iloc[fare,[6]].values[0]+1))

# Segregating X and y values
X = datset_NamingAddress.iloc[:, [3]].values
y = datset_NamingAddress.iloc[:,4]

X_submission = datset_submission_NamingAddress.iloc[:, [3]].values
index_list = []

X_test_visualize = X

# Encoding categorical data
# Encoding the Independent Variable Making all strings to values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#labelencoder_X_Title = LabelEncoder()
#X[:, 1] = labelencoder_X_Title.fit_transform(X[:, 1])

labelencoder_X_MF = LabelEncoder()
X[:, 0] = labelencoder_X_MF.fit_transform(X[:, 0])

#labelencoder_X_submission_Title = LabelEncoder()
#X_submission[:, 1] = labelencoder_X_submission_Title.fit_transform(X_submission[:, 1])

labelencoder_X_submission_MF = LabelEncoder()
X_submission[:, 0] = labelencoder_X_submission_MF.fit_transform(X_submission[:, 0])

# Encoding categorical data
# Encoding the Independent Variable  - disecting the equall classes seperate

onehotencoder_X = OneHotEncoder(categorical_features = [0])
X = onehotencoder_X.fit_transform(X).toarray()

onehotencoder_X_submission = OneHotEncoder(categorical_features = [0])
X_submission = onehotencoder_X_submission.fit_transform(X_submission).toarray()

# Splitting the dataset into the Training set and Test set
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
    
sc_X_train = StandardScaler()
X_train = sc_X_train.fit_transform(X_train)

sc_X_test = StandardScaler()
X_test = sc_X_test.fit_transform(X_test)

sc_X_submission = StandardScaler()
X_submission = sc_X_submission.fit_transform(X_submission)

# Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#regressor.fit(X, y)

# Predicting the test results
y_pred = regressor.predict(X_test)

# predicting variance score
from sklearn.metrics import explained_variance_score
varscore_Lin_reg = explained_variance_score(y_test, y_pred, sample_weight=None, multioutput='raw_values')

print('Linear Regression varscore :'+str(varscore_Lin_reg))

# Predicting r2 value
from sklearn.metrics import r2_score
r2_Lin_reg = r2_score(y_test, y_pred, sample_weight=None, multioutput='raw_values')

print('Linear Regression r2 :'+str(r2_Lin_reg))

# Fitting the SVR Model to the dataset
from sklearn.svm import SVR
KernelSVR_regressor = SVR(kernel = 'rbf') 
KernelSVR_regressor.fit(X_train, y_train)

# Predicting a new result
y_pred_svr = KernelSVR_regressor.predict(X_test)

# predicting variance score
from sklearn.metrics import explained_variance_score
varscore_KernelSVR = explained_variance_score(y_test, y_pred_svr, sample_weight=None, multioutput='uniform_average')

print('Kernel SVR varscore :'+str(varscore_KernelSVR))

# Predicting r2 value
from sklearn.metrics import r2_score
r2_KernelSVR = r2_score(y_test, y_pred_svr, sample_weight=None, multioutput='uniform_average')

print('Kernel SVR r2 :'+str(r2_KernelSVR))

# Fitting the DecisionTree Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
DecisionTree_regressor = DecisionTreeRegressor()
DecisionTree_regressor.fit(X_train, y_train)

# Predicting a new result
y_pred_DecisionTree = DecisionTree_regressor.predict(X_test)

# predicting r2 and variance score

varscore_DecisionTree = explained_variance_score(y_test, y_pred_DecisionTree, sample_weight=None, multioutput='uniform_average')
print('DecisionTree varscore :'+str(varscore_DecisionTree))
r2_DecisionTree = r2_score(y_test, y_pred_DecisionTree, sample_weight=None, multioutput='uniform_average')
print('DecisionTree r2 :'+str(r2_DecisionTree))

# Fitting the RandomForest Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
RandomForest_regressor = RandomForestRegressor(n_estimators=300)
RandomForest_regressor.fit(X_train, y_train)

# Predicting a new result
y_pred_RandomForest = RandomForest_regressor.predict(X_test)

# predicting r2 and variance score

varscore_RandomForest = explained_variance_score(y_test, y_pred_RandomForest, sample_weight=None, multioutput='uniform_average')
print('RandomForest varscore :'+str(varscore_RandomForest))
r2_RandomForest = r2_score(y_test, y_pred_RandomForest, sample_weight=None, multioutput='uniform_average')
print('RandomForest r2 :'+str(r2_RandomForest))

# Building the optimal model using backward elimination SL = 0.05, remove morethan 5%
import statsmodels.formula.api as sm
X_stats = np.append(arr = np.ones((len(X),1)).astype(int), values = X, axis = 1)
X_opt = X_stats[:, [0,1,2]]
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()