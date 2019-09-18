# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 10:16:58 2019

@author: admin
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Import Dataset
train_dataset = pd.read_csv('train.csv')
# Use train_dataset.isna().any() for checking columns for NaN in a columns
print('$$Dataset imported-success$$')

#Splitting X and y
X = train_dataset.iloc[:,1:-1]
y = train_dataset.iloc[:,-1]
y = np.reshape(np.ravel(y),(len(y),1))
print('$$Splitting X&y$$')

# Working on X
# Imputing and editring Values
# Encoding categorical data
# Encoding the Independent Variable Making all strings to values
X0to2 = X.iloc[:,0:3] #NoCh
X0to2 = X0to2.values
sc_X0to2 = StandardScaler()
X0to2 = sc_X0to2.fit_transform(X0to2)
print('$$X0to2-success$$')

X3 = X.iloc[:,3] #OHE
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

X4 = X.iloc[:,4] #OHE
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

X5 = X.iloc[:,5] #OHE
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

X6DOB = X.iloc[:,7]
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

X7 = X.iloc[:,8] # Imp, OHE
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

X8 = X.iloc[:,10] #OHE
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
print('$$X9-success$$')

X10 = X.iloc[:,13] #OHE
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
print('$$X10-success$$')

X11 = X.iloc[:,14] #OHE
X11  = X11.values.reshape((len(X11),1))
labelencoder_X11 = LabelEncoder()
labelencoder_X11 = labelencoder_X11.fit(X11)
X11 = labelencoder_X11.transform(X11)
X11 = X11.reshape((len(X11),1))
onehotencoder_X11 = OneHotEncoder(categorical_features = [0])
onehotencoder_X11 = onehotencoder_X11.fit(X11)
X11 = onehotencoder_X11.transform(X11).toarray()
X11 = pd.DataFrame(X11)
X11 = X11.values
sc_X11 = StandardScaler()
X11 = sc_X11.fit_transform(X11)
print('$$X11-success$$')

X12 = X.iloc[:,15] #OHE
X12  = X12.values.reshape((len(X12),1))
labelencoder_X12 = LabelEncoder()
labelencoder_X12 = labelencoder_X12.fit(X12)
X12 = labelencoder_X12.transform(X12)
X12 = X12.reshape((len(X12),1))
onehotencoder_X12 = OneHotEncoder(categorical_features = [0])
onehotencoder_X12 = onehotencoder_X12.fit(X12)
X12 = onehotencoder_X12.transform(X12).toarray()
X12 = pd.DataFrame(X12)
X12 = X12.values
sc_X12 = StandardScaler()
X12 = sc_X12.fit_transform(X12)
print('$$X12-success$$')

X13 = X.iloc[:,16] #OHE
X13  = X13.values.reshape((len(X13),1))
labelencoder_X13 = LabelEncoder()
labelencoder_X13 = labelencoder_X13.fit(X13)
X13 = labelencoder_X13.transform(X13)
X13 = X13.reshape((len(X13),1))
onehotencoder_X13 = OneHotEncoder(categorical_features = [0])
onehotencoder_X13 = onehotencoder_X13.fit(X13)
X13 = onehotencoder_X13.transform(X13).toarray()
X13 = pd.DataFrame(X13)
X13 = X13.values
sc_X13 = StandardScaler()
X13 = sc_X13.fit_transform(X13)
print('$$X13-success$$')

X14 = X.iloc[:,17] #OHE
X14  = X14.values.reshape((len(X14),1))
labelencoder_X14 = LabelEncoder()
labelencoder_X14 = labelencoder_X14.fit(X14)
X14 = labelencoder_X14.transform(X14)
X14 = X14.reshape((len(X14),1))
onehotencoder_X14 = OneHotEncoder(categorical_features = [0])
onehotencoder_X14 = onehotencoder_X14.fit(X14)
X14 = onehotencoder_X14.transform(X14).toarray()
X14 = pd.DataFrame(X14)
X14 = X14.values
sc_X14 = StandardScaler()
X14 = sc_X14.fit_transform(X14)
print('$$X14-success$$')

X15 = X.iloc[:,18] # NoCh
X15 = pd.DataFrame(X15)
X15 = X15.values
sc_X15 = StandardScaler()
X15 = sc_X15.fit_transform(X15)
print('$$X15-success$$')

X16 = X.iloc[:,19] #OHE
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

X17 = X.iloc[:,20:36] # NoCh
X17 = X17.values
sc_X17 = StandardScaler()
X17 = sc_X17.fit_transform(X17)
print('$$X17-success$$')

X18Time = X.iloc[:,36] # Edit, Put in months
X18 = np.zeros((len(X18Time),1))
for monConv in range(len(X18Time)):
    X18_V = X18Time.iloc[monConv,]
    if str(X18_V[6]) == 'm':
        X18[monConv] = int(X18_V[0])*12+int(X18_V[5])
    else:
        X18[monConv] = int(X18_V[0])*12+int(X18_V[5:7])
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
        X19[monConv] = int(X19_V[0])*12+int(X19_V[5])
    elif (str(X19_V[6]) != 'm') & (str(X19_V[1]) == 'y'):
        X19[monConv] = int(X19_V[0])*12+int(X19_V[5:7])
    elif (str(X19_V[7]) != 'm') & (str(X19_V[1]) != 'y'):
        X19[monConv] = int(X19_V[0:2])*12+int(X19_V[6:8])
    else:
        X19[monConv] = int(X19_V[0:2])*12+int(X19_V[6])
X19 = pd.DataFrame(X19)
X19 = X19.values
sc_X19 = StandardScaler()
X19 = sc_X19.fit_transform(X19)
print('$$X19-success$$')

X20 = X.iloc[:,38] #NoCh
X20 = pd.DataFrame(X20)
X20 = X20.values
sc_X20 = StandardScaler()
X20 = sc_X20.fit_transform(X20)
print('$$X20-success$$')

# Concating X
X_b4_edit = X.copy()
X = np.concatenate((X0to2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20),axis=1)
#X = pd.concat([X0to2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20],axis = 1)
print('$$X Concate-success$$')

#Normalising all Values
#sc_X = StandardScaler()
#X = sc_X.fit_transform(X)




SVM_classifier = SVC(kernel = 'linear', random_state = 0)
SVM_classifier.fit(X, y)
print('$$SVM-Linear_Classifier-success$$')'''











