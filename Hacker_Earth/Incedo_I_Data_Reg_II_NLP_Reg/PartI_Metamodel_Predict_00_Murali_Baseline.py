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


dataset = pd.read_csv('train_Murali.csv')
test_dataset = pd.read_csv('test_Murali.csv')


X = dataset.iloc[:,:-1].values
X = pd.DataFrame(X)
X_RealTest = test_dataset.iloc[:,:].values
X_RealTest= pd.DataFrame(X_RealTest)
y = dataset.iloc[:,-1].values



X1 = X.iloc[:,1] #OHE
X1  = X1.values.reshape((len(X1),1))
labelencoder_X1 = LabelEncoder()
labelencoder_X1 = labelencoder_X1.fit(X1)
X1 = labelencoder_X1.transform(X1)
X1 = X1.reshape((len(X1),1))
onehotencoder_X1 = OneHotEncoder(categorical_features = [0])
onehotencoder_X1 = onehotencoder_X1.fit(X1)
X1 = onehotencoder_X1.transform(X1).toarray()
# test
X1_test = X_RealTest.iloc[:,1] #OHE
X1_test  = X1_test.values.reshape((len(X1_test),1))
labelencoder_X1_test = LabelEncoder()
labelencoder_X1_test = labelencoder_X1_test.fit(X1_test)
X1_test = labelencoder_X1_test.transform(X1_test)
X1_test = X1_test.reshape((len(X1_test),1))
onehotencoder_X1_test = OneHotEncoder()
onehotencoder_X1_test = onehotencoder_X1_test.fit(X1_test)
X1_test = onehotencoder_X1_test.transform(X1_test).toarray()


X2 = X.iloc[:,2] #OHE
X2  = X2.values.reshape((len(X2),1))
labelencoder_X2 = LabelEncoder()
labelencoder_X2 = labelencoder_X2.fit(X2)
X2 = labelencoder_X2.transform(X2)
X2 = X2.reshape((len(X2),1))
onehotencoder_X2 = OneHotEncoder(categorical_features = [0])
onehotencoder_X2 = onehotencoder_X2.fit(X2)
X2 = onehotencoder_X2.transform(X2).toarray()
# test
X2_test = X_RealTest.iloc[:,2] #OHE
X2_test  = X2_test.values.reshape((len(X2_test),1))
labelencoder_X2_test = LabelEncoder()
labelencoder_X2_test = labelencoder_X2_test.fit(X2_test)
X2_test = labelencoder_X2_test.transform(X2_test)
X2_test = X2_test.reshape((len(X2_test),1))
onehotencoder_X2_test = OneHotEncoder(categorical_features = [0])
onehotencoder_X2_test = onehotencoder_X2_test.fit(X2_test)
X2_test = onehotencoder_X2_test.transform(X2_test).toarray()

X3 = X.iloc[:,3] #OHE
X3  = X3.values.reshape((len(X3),1))
labelencoder_X3 = LabelEncoder()
labelencoder_X3 = labelencoder_X3.fit(X3)
X3 = labelencoder_X3.transform(X3)
X3 = X3.reshape((len(X3),1))
onehotencoder_X3 = OneHotEncoder(categorical_features = [0])
onehotencoder_X3 = onehotencoder_X3.fit(X3)
X3 = onehotencoder_X3.transform(X3).toarray()
# test
X3_test = X_RealTest.iloc[:,3] #OHE
X3_test  = X3_test.values.reshape((len(X3_test),1))
labelencoder_X3_test = LabelEncoder()
labelencoder_X3_test = labelencoder_X3_test.fit(X3_test)
X3_test = labelencoder_X3_test.transform(X3_test)
X3_test = X3_test.reshape((len(X3_test),1))
onehotencoder_X3_test = OneHotEncoder(categorical_features = [0])
onehotencoder_X3_test = onehotencoder_X3_test.fit(X3_test)
X3_test = onehotencoder_X3_test.transform(X3_test).toarray()

X4 = X.iloc[:,4] #OHE
X4  = X4.values.reshape((len(X4),1))
labelencoder_X4 = LabelEncoder()
labelencoder_X4 = labelencoder_X4.fit(X4)
X4 = labelencoder_X4.transform(X4)
X4 = X4.reshape((len(X4),1))
onehotencoder_X4 = OneHotEncoder(categorical_features = [0])
onehotencoder_X4 = onehotencoder_X4.fit(X4)
X4 = onehotencoder_X4.transform(X4).toarray()
# test
X4_test = X_RealTest.iloc[:,4] #OHE
X4_test  = X4_test.values.reshape((len(X4_test),1))
labelencoder_X4_test = LabelEncoder()
labelencoder_X4_test = labelencoder_X4_test.fit(X4_test)
X4_test = labelencoder_X4_test.transform(X4_test)
X4_test = X4_test.reshape((len(X4_test),1))
onehotencoder_X4_test = OneHotEncoder(categorical_features = [0])
onehotencoder_X4_test = onehotencoder_X4_test.fit(X4_test)
X4_test = onehotencoder_X4_test.transform(X4_test).toarray()

X5 = X.iloc[:,5] #OHE
X5  = X5.values.reshape((len(X5),1))
labelencoder_X5 = LabelEncoder()
labelencoder_X5 = labelencoder_X5.fit(X5)
X5 = labelencoder_X5.transform(X5)
X5 = X5.reshape((len(X5),1))
onehotencoder_X5 = OneHotEncoder(categorical_features = [0])
onehotencoder_X5 = onehotencoder_X5.fit(X5)
X5 = onehotencoder_X5.transform(X5).toarray()
# test
X5_test = X_RealTest.iloc[:,5] #OHE
X5_test  = X5_test.values.reshape((len(X5_test),1))
labelencoder_X5_test = LabelEncoder()
labelencoder_X5_test = labelencoder_X5_test.fit(X5_test)
X5_test = labelencoder_X5_test.transform(X5_test)
X5_test = X5_test.reshape((len(X5_test),1))
onehotencoder_X5_test = OneHotEncoder(categorical_features = [0])
onehotencoder_X5_test = onehotencoder_X5_test.fit(X5_test)
X5_test = onehotencoder_X5_test.transform(X5_test).toarray()

from sklearn.preprocessing import StandardScaler
X6 = X.iloc[:,6:13]
X6 = pd.DataFrame(X6)
X6 = X6.values
sc_X6 = StandardScaler()
X6 = sc_X6.fit_transform(X6)
# Concating X
X_b4_edit = X.copy()

X6_test = X_RealTest.iloc[:,6:13]
X6_test = pd.DataFrame(X6_test)
X6_test = X6_test.values
sc_X6_test = StandardScaler()
X6_test = sc_X6.fit_transform(X6_test)



X_Concat = np.concatenate((X1,X2,X3,X4,X5,X6),axis=1)
X_Concat_test = np.concatenate((X1_test,X2_test,X3_test,X4_test,X5_test,X6_test),axis=1)


X_Concat = pd.DataFrame(X_Concat)
X_Concat_test = pd.DataFrame(X_Concat_test)

X_train, X_test, y_train, y_test = train_test_split(X_Concat, y, test_size = 0.25)

print('$$Splitting X&y-success$$')

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300)
regressor.fit(X_Concat,y)

'''y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
r2_scorepred = r2_score(y_test, y_pred, sample_weight=None)
print(r2_scorepred)

from sklearn.metrics import mean_absolute_error
mean_absolute_errorpred = mean_absolute_error(y_test, y_pred, sample_weight=None)
print(mean_absolute_errorpred)'''

# y_pred = regressor.predict(X_test)
y_predreal = regressor.predict(X_Concat_test)

print('$$Prediction-success$$')