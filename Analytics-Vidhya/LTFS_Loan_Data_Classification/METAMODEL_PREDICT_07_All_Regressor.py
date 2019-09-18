# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 10:16:58 2019

@author: admin
"""

import pandas as pd
import numpy as np
import timeit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error

start = timeit.default_timer()
# Import Dataset
train_dataset = pd.read_csv('train.csv')
Column_Names = train_dataset.columns
train_dataset = train_dataset.dropna(axis=0,subset=['Employment.Type'])
test_dataset = pd.read_csv('test.csv')
train_dataset = pd.concat([train_dataset,test_dataset],axis =0,ignore_index=True)
train_dataset = train_dataset.reindex_axis(Column_Names,axis=1)
print('$$Dataset imported-success$$')

#Data Cleaning
X18_pp=train_dataset.iloc[:,18]
X18_pp=X18_pp.values
y_pp=train_dataset.iloc[:,-1]
y_pp=y_pp.values
rowsDropList = []
for ppcheck in range(233154):
    if X18_pp[ppcheck] == 1 and y_pp[ppcheck] == 1:
        rowsDropList.append(ppcheck) 
print('$$Data cleaning-success$$')
train_dataset = train_dataset.drop(rowsDropList)

X20_ni=train_dataset.iloc[:,-2]
X20_ni=X20_ni.values
rowsDropList = []
for nicheck in range(len(X20_ni)):
    if X20_ni[nicheck] > 1:
        train_dataset.iloc[nicheck,-2] = 1 
index = pd.Index(train_dataset.iloc[:,-2])
print(index.value_counts())
print('$$Data cleaning-success$$')
#Splitting X and y
X = train_dataset.iloc[:,1:-1]
y = train_dataset.iloc[:225422,-1]
y = np.reshape(np.ravel(y),(len(y),1))
#X_test = test_dataset.iloc[:,1:]
#y = train_dataset.iloc[:,-1]
#y = np.reshape(np.ravel(y),(len(y),1))
print('$$Splitting X&y$$')

# Working on X
# Imputing and editring Values
# Encoding categorical data
# Encoding the Independent Variable Making all strings to values
X2 = X.iloc[:,2] #NoCh #ltv
X2 = pd.DataFrame(X2)
X2 = X2.values
sc_X2 = StandardScaler()
X2 = sc_X2.fit_transform(X2)
print('$$X2-success$$')
'''
X3 = X.iloc[:,3] #OHE #Branch Id
X3  = X3.values.reshape((len(X3),1))
labelencoder_X3 = LabelEncoder()
labelencoder_X3 = labelencoder_X3.fit(X3)
X3 = labelencoder_X3.transform(X3)
X3 = X3.reshape((len(X3),1))
onehotencoder_X3 = OneHotEncoder(categorical_features = [0])
onehotencoder_X3 = onehotencoder_X3.fit(X3)
X3 = onehotencoder_X3.transform(X3).toarray()
X3 = pd.DataFrame(X3)
X3 = X3.values
sc_X3 = StandardScaler()
X3 = sc_X3.fit_transform(X3)
print('$$X3-success$$')

X4 = X.iloc[:,4] #OHE #Supplier id
X4  = X4.values.reshape((len(X4),1))
labelencoder_X4 = LabelEncoder()
labelencoder_X4 = labelencoder_X4.fit(X4)
X4 = labelencoder_X4.transform(X4)
X4 = X4.reshape((len(X4),1))
onehotencoder_X4 = OneHotEncoder(categorical_features = [0])
onehotencoder_X4 = onehotencoder_X4.fit(X4)
X4 = onehotencoder_X4.transform(X4).toarray()
X4 = pd.DataFrame(X4)
X4 = X4.values
sc_X4 = StandardScaler()
X4 = sc_X4.fit_transform(X4)
print('$$X4-success$$')

X5 = X.iloc[:,5] #OHE # Manufacturer ID
X5  = X5.values.reshape((len(X5),1)) 
labelencoder_X5 = LabelEncoder()
labelencoder_X5 = labelencoder_X5.fit(X5)
X5 = labelencoder_X5.transform(X5)
X5 = X5.reshape((len(X5),1))
onehotencoder_X5 = OneHotEncoder(categorical_features = [0])
onehotencoder_X5 = onehotencoder_X5.fit(X5)
X5 = onehotencoder_X5.transform(X5).toarray()
X5 = pd.DataFrame(X5)
X5 = X5.values
sc_X5 = StandardScaler()
X5 = sc_X5.fit_transform(X5)
print('$$X5-success$$')
'''
X6DOB = X.iloc[:,7] # Getting Age
X6DisbDate = X.iloc[:,9] # Get No. of months data
X6 = np.zeros((len(X6DOB),1))
for ageCalc in range(len(X6DOB)):
    X6DOB_V = X6DOB.iloc[ageCalc,]
    X6DOB_V = int(X6DOB_V[6:8])*12 + int(X6DOB_V[3:5]) + (int(X6DOB_V[0:2])/30)
    X6DisbDate_V = X6DisbDate.iloc[ageCalc,]
    X6DisbDate_V = (100+int(X6DisbDate_V[6:8]))*12 + int(X6DisbDate_V[3:5]) + (int(X6DisbDate_V[0:2])/30)
    X6[ageCalc] = (X6DisbDate_V-X6DOB_V)
X6 = pd.DataFrame(X6)
X6 = X6.values
sc_X6 = StandardScaler()
X6 = sc_X6.fit_transform(X6)
print('$$X6-success$$')

X7 = X.iloc[:,8] # Imp, OHE # Employment Type
#Imputing - Taking care of missing data
X7 = X7.fillna('Not Provided')
X7  = X7.values.reshape((len(X7),1))
labelencoder_X7 = LabelEncoder()
labelencoder_X7 = labelencoder_X7.fit(X7)
X7 = labelencoder_X7.transform(X7)
X7 = X7.reshape((len(X7),1))
onehotencoder_X7 = OneHotEncoder(categorical_features = [0])
onehotencoder_X7 = onehotencoder_X7.fit(X7)
X7 = onehotencoder_X7.transform(X7).toarray()
X7 = pd.DataFrame(X7)
X7 = X7.values
sc_X7 = StandardScaler()
X7 = sc_X7.fit_transform(X7)
print('$$X7-success$$')

'''X8 = X.iloc[:,10] #OHE
X8  = X8.values.reshape((len(X8),1))
labelencoder_X8 = LabelEncoder()
labelencoder_X8 = labelencoder_X8.fit(X8)
X8 = labelencoder_X8.transform(X8)
X8 = X8.reshape((len(X8),1))
onehotencoder_X8 = OneHotEncoder(categorical_features = [0])
onehotencoder_X8 = onehotencoder_X8.fit(X8)
X8 = onehotencoder_X8.transform(X8).toarray()
X8 = pd.DataFrame(X8)
X8 = X8.values
sc_X8 = StandardScaler()
X8 = sc_X8.fit_transform(X8)
print('$$X8-success$$')

X9 = X.iloc[:,11] #OHE
X9  = X9.values.reshape((len(X9),1))
labelencoder_X9 = LabelEncoder()
labelencoder_X9 = labelencoder_X9.fit(X9)
X9 = labelencoder_X9.transform(X9)
X9 = X9.reshape((len(X9),1))
onehotencoder_X9 = OneHotEncoder(categorical_features = [0])
onehotencoder_X9 = onehotencoder_X9.fit(X9)
X9 = onehotencoder_X9.transform(X9).toarray()
X9 = pd.DataFrame(X9)
X9 = X9.values
sc_X9 = StandardScaler()
X9 = sc_X9.fit_transform(X9)
print('$$X9-success$$')'''

X11 = X.iloc[:,14] #OHE#Pan Card
X12 = X.iloc[:,15] #OHE # Voter_id
X13 = X.iloc[:,16] #OHE # Driving License
X11_13 = [X11.iloc[i] + X12.iloc[i] +X13.iloc[i] for i in range(len(X11))]
X11_13 = pd.DataFrame(X11_13)
X11_13 = X11_13.values
sc_X11_13 = StandardScaler()
X11_13 = sc_X11_13.fit_transform(X11_13)
print('$$X11_13-success$$')

X14 = X.iloc[:,17] #OHE # Passport
X14 = X14.reshape((len(X14),1))
onehotencoder_X14 = OneHotEncoder(categorical_features = [0])
onehotencoder_X14 = onehotencoder_X14.fit(X14)
X14 = onehotencoder_X14.transform(X14).toarray()
X14 = pd.DataFrame(X14)
X14 = X14.values
sc_X14 = StandardScaler()
X14 = sc_X14.fit_transform(X14)
print('$$X14-success$$')

'''X15 = X.iloc[:,18] # NoCh
X15 = pd.DataFrame(X15)
X15 = X15.values
sc_X15 = StandardScaler()
X15 = sc_X15.fit_transform(X15)
print('$$X15-success$$')
'''
X16 = X.iloc[:,19] #OHE # Credit Rating
X16  = X16.values.reshape((len(X16),1))
labelencoder_X16 = LabelEncoder()
labelencoder_X16 = labelencoder_X16.fit(X16)
X16 = labelencoder_X16.transform(X16)
X16 = X16.reshape((len(X16),1))
onehotencoder_X16 = OneHotEncoder(categorical_features = [0])
onehotencoder_X16 = onehotencoder_X16.fit(X16)
X16 = onehotencoder_X16.transform(X16).toarray()
X16 = pd.DataFrame(X16)
X16 = X16.values
sc_X16 = StandardScaler()
X16 = sc_X16.fit_transform(X16)
print('$$X16-success$$')
'''
X17 = X.iloc[:,[20,22,23,28,34,35]] # NoCh
X17 = X17.values
sc_X17 = StandardScaler()
X17 = sc_X17.fit_transform(X17)
print('$$X17-success$$')

X18Time = X.iloc[:,36] # Edit, Put in months
X18 = np.zeros((len(X18Time),1))
for mon18Conv in range(len(X18Time)):
    X18_V = X18Time.iloc[mon18Conv,]
    if (str(X18_V[6]) == 'm') & (str(X18_V[1]) == 'y'):
        X18[mon18Conv] = int(X18_V[0])*12+int(X18_V[5])
    elif (str(X18_V[6]) != 'm') & (str(X18_V[1]) == 'y'):
        X18[mon18Conv] = int(X18_V[0])*12+int(X18_V[5:7])
    elif (str(X18_V[7]) != 'm') & (str(X18_V[1]) != 'y'):
        X18[mon18Conv] = int(X18_V[0:2])*12+int(X18_V[6:8])
    else:
        X18[mon18Conv] = int(X18_V[0:2])*12+int(X18_V[6])
X18 = pd.DataFrame(X18)
X18 = X18.values
sc_X18 = StandardScaler()
X18 = sc_X18.fit_transform(X18)
print('$$X18-success$$')

X19Time = X.iloc[:,37] # Edit, Put in months
X19 = np.zeros((len(X19Time),1))
for mon19Conv in range(len(X19Time)):
    X19_V = X19Time.iloc[mon19Conv,]
    if (str(X19_V[6]) == 'm') & (str(X19_V[1]) == 'y'):
        X19[mon19Conv] = int(X19_V[0])*12+int(X19_V[5])
    elif (str(X19_V[6]) != 'm') & (str(X19_V[1]) == 'y'):
        X19[mon19Conv] = int(X19_V[0])*12+int(X19_V[5:7])
    elif (str(X19_V[7]) != 'm') & (str(X19_V[1]) != 'y'):
        X19[mon19Conv] = int(X19_V[0:2])*12+int(X19_V[6:8])
    else:
        X19[mon19Conv] = int(X19_V[0:2])*12+int(X19_V[6])
X19 = pd.DataFrame(X19)
X19 = X19.values
sc_X19 = StandardScaler()
X19 = sc_X19.fit_transform(X19)
print('$$X19-success$$')
'''
X20 = X.iloc[:,38] #NoCh
X20 = pd.DataFrame(X20)
X20 = X20.values
sc_X20 = StandardScaler()
X20 = sc_X20.fit_transform(X20)
print('$$X20-success$$')

# Concating 
X_b4_edit = X.copy()
X_train_test = np.concatenate((X2,X7,X11_13,X14,X16,X20),axis=1)
X = X_train_test[:225422,:]
print('$$X Concate-success$$')

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
print('$$Splitting X&y-success$$')

#Regressor
# Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
Linear_regressor = LinearRegression()
Linear_regressor.fit(X_train, y_train)

# Predicting the test results
y_pred = Linear_regressor.predict(X_test)
y_pred = pd.DataFrame(y_pred)
y_test = pd.DataFrame(y_test)

print('Linear Regression')
LinReg_r2_score = r2_score(y_test, y_pred, sample_weight=None)
print(LinReg_r2_score)

LinReg_mean_absolute_error = mean_absolute_error(y_test, y_pred, sample_weight=None)
print(LinReg_mean_absolute_error)

# Fitting the Decision Tree Model to the dataset
from sklearn.tree import DecisionTreeRegressor
DT_regressor = DecisionTreeRegressor()
DT_regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = DT_regressor.predict(X_test)
y_pred.reshape(len(y_pred),1)

print('DT Regressor')
DT_r2_score = r2_score(y_test, y_pred, sample_weight=None)
print(DT_r2_score)

DT_mean_absolute_error = mean_absolute_error(y_test, y_pred, sample_weight=None)
print(DT_mean_absolute_error)

# Fitting the Random Forest Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
RF_regressor = RandomForestRegressor(n_estimators=300, random_state=0)
RF_regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = RF_regressor.predict(X_test)

print('RF Regressor')
RF_r2_score = r2_score(y_test, y_pred, sample_weight=None)
print(RF_r2_score)

RF_mean_absolute_error = mean_absolute_error(y_test, y_pred, sample_weight=None)
print(RF_mean_absolute_error)

'''# Getting test submission values
# Regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X,y)

X_s = X_train_test[225422:,:]
y_pred_s = DT_regressor.predict(X_s)'''

#-------------------------------------------------------------------------------------------#
'''X10 = X.iloc[:,13] #OHE #Aadhar
X10  = X10.values.reshape((len(X10),1))
labelencoder_X10 = LabelEncoder()
labelencoder_X10 = labelencoder_X10.fit(X10)
X10 = labelencoder_X10.transform(X10)
X10 = X10.reshape((len(X10),1))
onehotencoder_X10 = OneHotEncoder(categorical_features = [0])
onehotencoder_X10 = onehotencoder_X10.fit(X10)
X10 = onehotencoder_X10.transform(X10).toarray()
X10 = pd.DataFrame(X10)
X10 = X10.values
sc_X10 = StandardScaler()
X10 = sc_X10.fit_transform(X10)
print('$$X10-success$$')'''

'''#logreg classifier
LogReg_classifier = LogisticRegression()
LogReg_classifier.fit(X_train, y_train)

# Prediction
y_pred = LogReg_classifier.predict(X_test)
print('$$LogReg Prediction-success$$')

from sklearn.metrics import accuracy_score   
Accuracy_RanForest =accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print('LogReg')
print(Accuracy_RanForest*100)

from sklearn.metrics import confusion_matrix
cnf_mat = confusion_matrix(y_test, y_pred, labels=[0,1], sample_weight=None)
print('LogReg')
print(cnf_mat)'''

'''# Pickling process
import pickle

with open('V:/2019/WIP/Analytics_Vidhya_LTFS/METAMODEL_01_RanFor100.py', 'wb') as METAMODEL:
    pickle.dump(RanForest_classifier, METAMODEL)
print('$$Pickled - success$$')

stop = timeit.default_timer()
print('Time: ', stop - start)

# Fitting SVM on the dataset
from sklearn.svm import SVR
SVR_regressor = SVR(kernel = 'rbf') 
SVR_regressor.fit(X, y)

# Predicting the test results
y_pred = SVR_regressor.predict(X_test)

print('SVR Regressor')
r2_score = r2_score(y_test, y_pred, sample_weight=None)
print(r2_score)

mean_absolute_error = mean_absolute_error(y_test, y_pred, sample_weight=None)
print(mean_absolute_error)'''

'''# Fitting polynomial regression on the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
poly_reg.fit(X_train,y_train)

# Predicting the test results
y_pred = poly_reg.predict(X_test)

print('Polynomial Regression')
r2_score = r2_score(y_test, y_pred, sample_weight=None)
print(r2_score)

mean_absolute_error = mean_absolute_error(y_test, y_pred, sample_weight=None)
print(mean_absolute_error)'''








