# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 07:24:50 2019

@author: SJMB
"""



import pandas as pd
import numpy as np
import timeit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score,mean_absolute_error


train_dataset = pd.read_csv('train_Murali.csv')
Column_Names = train_dataset.columns
test_dataset = pd.read_csv('test_Murali.csv')
train_dataset = pd.concat([train_dataset,test_dataset],axis =0)
train_dataset = train_dataset.reindex_axis(Column_Names,axis=1)
print('$$Dataset imported-success$$')


X = train_dataset.iloc[:,:-1].values
X = pd.DataFrame(X)
y = train_dataset.iloc[:,-1].values
print("$$X,y Created$$")

import matplotlib.pyplot as plt

plt.matshow(X.corr())
plt.show()



'''X1 = X.iloc[:,1] #OHE
X1  = X1.values.reshape((len(X1),1))
labelencoder_X1 = LabelEncoder()
labelencoder_X1 = labelencoder_X1.fit(X1)
X1 = labelencoder_X1.transform(X1)
X1 = X1.reshape((len(X1),1))
onehotencoder_X1 = OneHotEncoder(categorical_features = [0])
onehotencoder_X1 = onehotencoder_X1.fit(X1)
X1 = onehotencoder_X1.transform(X1).toarray()
print("$$X1 Created$$")


X2 = X.iloc[:,2] #OHE
X2  = X2.values.reshape((len(X2),1))
labelencoder_X2 = LabelEncoder()
labelencoder_X2 = labelencoder_X2.fit(X2)
X2 = labelencoder_X2.transform(X2)
X2 = X2.reshape((len(X2),1))
onehotencoder_X2 = OneHotEncoder(categorical_features = [0])
onehotencoder_X2 = onehotencoder_X2.fit(X2)
X2 = onehotencoder_X2.transform(X2).toarray()
print("$$X2 Created$$")

X3 = X.iloc[:,3] #OHE
X3  = X3.values.reshape((len(X3),1))
labelencoder_X3 = LabelEncoder()
labelencoder_X3 = labelencoder_X3.fit(X3)
X3 = labelencoder_X3.transform(X3)
X3 = X3.reshape((len(X3),1))
onehotencoder_X3 = OneHotEncoder(categorical_features = [0])
onehotencoder_X3 = onehotencoder_X3.fit(X3)
X3 = onehotencoder_X3.transform(X3).toarray()
print("$$X3 Created$$")

X4 = X.iloc[:,4] #OHE
X4  = X4.values.reshape((len(X4),1))
labelencoder_X4 = LabelEncoder()
labelencoder_X4 = labelencoder_X4.fit(X4)
X4 = labelencoder_X4.transform(X4)
X4 = X4.reshape((len(X4),1))
onehotencoder_X4 = OneHotEncoder(categorical_features = [0])
onehotencoder_X4 = onehotencoder_X4.fit(X4)
X4 = onehotencoder_X4.transform(X4).toarray()
print("$$X4 Created$$")

X5 = X.iloc[:,5] #OHE
X5  = X5.values.reshape((len(X5),1))
labelencoder_X5 = LabelEncoder()
labelencoder_X5 = labelencoder_X5.fit(X5)
X5 = labelencoder_X5.transform(X5)
X5 = X5.reshape((len(X5),1))
onehotencoder_X5 = OneHotEncoder(categorical_features = [0])
onehotencoder_X5 = onehotencoder_X5.fit(X5)
X5 = onehotencoder_X5.transform(X5).toarray()
print("$$X5 Created$$")

X6 = X.iloc[:,6] #OHE #Year
X6  = X6.values.reshape((len(X6),1))
labelencoder_X6 = LabelEncoder()
labelencoder_X6 = labelencoder_X6.fit(X6)
X6 = labelencoder_X6.transform(X6)
X6 = X6.reshape((len(X6),1))
onehotencoder_X6 = OneHotEncoder(categorical_features = [0])
onehotencoder_X6 = onehotencoder_X6.fit(X6)
X6 = onehotencoder_X6.transform(X6).toarray()
print("$$X6 Created$$")

X7 = X.iloc[:,7] #SampleSize
X7 = pd.DataFrame(X6)
X7 = X7.values
sc_X7 = StandardScaler()
X7 = sc_X7.fit_transform(X6)
print("$$X7 Created$$")

X8 = X.iloc[:,8] #OHE #Subtopic
X8  = X8.values.reshape((len(X8),1))
labelencoder_X8 = LabelEncoder()
labelencoder_X8 = labelencoder_X8.fit(X8)
X8 = labelencoder_X8.transform(X8)
X8 = X8.reshape((len(X8),1))
onehotencoder_X8 = OneHotEncoder(categorical_features = [0])
onehotencoder_X8 = onehotencoder_X8.fit(X8)
X8 = onehotencoder_X8.transform(X8).toarray()
print("$$X8 Created$$")

X9 = X.iloc[:,9] #OHE #Grade
X9  = X9.values.reshape((len(X9),1))
labelencoder_X9 = LabelEncoder()
labelencoder_X9 = labelencoder_X9.fit(X9)
X9 = labelencoder_X9.transform(X9)
X9 = X9.reshape((len(X9),1))
onehotencoder_X9 = OneHotEncoder(categorical_features = [0])
onehotencoder_X9 = onehotencoder_X9.fit(X9)
X9 = onehotencoder_X9.transform(X9).toarray()
print("$$X9 Created$$")

X10 = X.iloc[:,10] #OHE #StratID1
X10  = X10.values.reshape((len(X10),1))
labelencoder_X10 = LabelEncoder()
labelencoder_X10 = labelencoder_X10.fit(X10)
X10 = labelencoder_X10.transform(X10)
X10 = X10.reshape((len(X10),1))
onehotencoder_X10 = OneHotEncoder(categorical_features = [0])
onehotencoder_X10 = onehotencoder_X10.fit(X10)
X10 = onehotencoder_X10.transform(X10).toarray()
print("$$X10 Created$$")

X11 = X.iloc[:,11] #OHE #StratID2
X11  = X11.values.reshape((len(X11),1))
labelencoder_X11 = LabelEncoder()
labelencoder_X11 = labelencoder_X11.fit(X11)
X11 = labelencoder_X11.transform(X11)
X11 = X11.reshape((len(X11),1))
onehotencoder_X11 = OneHotEncoder(categorical_features = [0])
onehotencoder_X11 = onehotencoder_X11.fit(X11)
X11 = onehotencoder_X11.transform(X11).toarray()
print("$$X11 Created$$")

X12 = X.iloc[:,12] #OHE #StratID3
X12  = X12.values.reshape((len(X12),1))
labelencoder_X12 = LabelEncoder()
labelencoder_X12 = labelencoder_X12.fit(X12)
X12 = labelencoder_X12.transform(X12)
X12 = X12.reshape((len(X12),1))
onehotencoder_X12 = OneHotEncoder(categorical_features = [0])
onehotencoder_X12 = onehotencoder_X12.fit(X12)
X12 = onehotencoder_X12.transform(X12).toarray()
print("$$X12 Created$$")

X_Concat = np.concatenate((X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12),axis=1)
X = X_Concat[:55399,:]
y = y[:55399]
X_s = X_Concat[55399:,:]

X_Concat = pd.DataFrame(X)
print("$$X Concat done$$")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
print('$$Splitting X&y-success$$')

#Regressor
# Fitting multiple linear regression to the training set
'''from sklearn.linear_model import LinearRegression
Linear_regressor = LinearRegression()
Linear_regressor.fit(X_train, y_train)

# Predicting the test results
# Random Forest 400
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=400)
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

print('Random Forest 400')
r2_scorepred = r2_score(y_test, y_pred, sample_weight=None)
print(r2_scorepred)

mean_absolute_errorpred = mean_absolute_error(y_test, y_pred, sample_weight=None)
print(mean_absolute_errorpred)'''

#MLP Regresor
from sklearn.neural_network import MLPRegressor
ANN_Regressor = MLPRegressor(hidden_layer_sizes=(150,100,50,10), activation='relu', solver='adam', alpha=0.5, batch_size='auto', learning_rate='invscaling', learning_rate_init=0.001, power_t=0.5, max_iter=400, shuffle=True, random_state=None, tol=0.0001, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
ANN_Regressor.fit(X_train,y_train)

y_pred = ANN_Regressor.predict(X_test)

for i in range(len(y_pred)):
    if y_pred[i] < 0:
        y_pred[i] = 0
    if y_pred[i] > 100:
        y_pred[i] = 100

print('MLP Regressor')
MLP_r2_score = r2_score(y_test, y_pred, sample_weight=None)
print(MLP_r2_score)

MLP_mean_absolute_error = mean_absolute_error(y_test, y_pred, sample_weight=None)
print(MLP_mean_absolute_error)

# Test result submission
ANN_Regressor = MLPRegressor(hidden_layer_sizes=(100,20), activation='relu', solver='adam', alpha=10, batch_size='auto', learning_rate='invscaling', learning_rate_init=0.001, power_t=0.5, max_iter=400, shuffle=True, random_state=None, tol=0.0001, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
ANN_Regressor.fit(X,y)

y_s = ANN_Regressor.predict(X_s)

for i in range(len(y_s)):
    if y_s[i] < 0:
        y_s[i] = 0
    if y_s[i] > 100:
        y_s[i] = 100

print('$$Final test Prediction-success$$')'''