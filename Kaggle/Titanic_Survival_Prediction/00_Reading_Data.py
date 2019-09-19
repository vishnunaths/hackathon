# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 15:15:19 2019

@author: kzt9qh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [2,4,5]].values
y = dataset.iloc[:,1]
dataset_submission = pd.read_csv('test.csv')
X_submission = dataset_submission.iloc[:, [1,3,4]].values
Accuracy_Classifier=pd.DataFrame(columns=['KNN', 'LogReg', 'SVM', 'KernelSVM', 'NB', 'DecTree', 'RanForest'])
Accuracy_Classifier[0,0] = 'KNN'
Accuracy_Classifier[0,1] = 'LogReg'
Accuracy_Classifier[0,2] = 'SVM'
Accuracy_Classifier[0,3] = 'KernelSVM'
Accuracy_Classifier[0,4] = 'NB'
Accuracy_Classifier[0,5] = 'DecTree'
Accuracy_Classifier[0,6] = 'RanForest'

# Encoding categorical data
# Encoding the Independent Variable Making all strings to values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

labelencoder_X_submission = LabelEncoder()
X_submission[:, 1] = labelencoder_X_submission.fit_transform(X_submission[:, 1])

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)
imputer_submission = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_submission = imputer_submission.fit(X_submission)
X_submission = imputer_submission.transform(X_submission)

# Encoding categorical data
# Encoding the Independent Variable  - disecting the equall classes seperate

onehotencoder_X = OneHotEncoder(categorical_features = [0,1])
X = onehotencoder_X.fit_transform(X).toarray()

onehotencoder_X_submission = OneHotEncoder(categorical_features = [0,1])
X_submission = onehotencoder_X_submission.fit_transform(X_submission).toarray()

for NumberOfRuns in range(0,100):
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X_train = StandardScaler()
    X_train = sc_X_train.fit_transform(X_train)

    sc_X_test = StandardScaler()
    X_test = sc_X_test.fit_transform(X_test)

    sc_X_submission = StandardScaler()
    X_submission = sc_X_submission.fit_transform(X_submission)

    # Fitting Classifier to the Training set
    from sklearn.neighbors import KNeighborsClassifier
    KNN_classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2)
    KNN_classifier.fit(X_train, y_train)
    from sklearn.linear_model import LogisticRegression
    LogReg_classifier = LogisticRegression(random_state = 0)
    LogReg_classifier.fit(X_train, y_train)
    from sklearn.svm import SVC
    SVM_classifier = SVC(kernel = 'linear', random_state = 0)
    SVM_classifier.fit(X_train, y_train)
    from sklearn.svm import SVC
    KernelSVM_classifier = SVC(kernel = 'rbf', random_state = 0)
    KernelSVM_classifier.fit(X_train, y_train)
    from sklearn.naive_bayes import GaussianNB
    NB_classifier = GaussianNB()
    NB_classifier.fit(X_train, y_train)
    from sklearn.tree import DecisionTreeClassifier
    DecTree_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    DecTree_classifier.fit(X_train, y_train)
    from sklearn.ensemble import RandomForestClassifier
    RanForest_classifier = RandomForestClassifier(n_estimators = 25, criterion = 'entropy', random_state = 0)
    RanForest_classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred_KNN = KNN_classifier.predict(X_test)
    y_pred_LogReg = LogReg_classifier.predict(X_test)
    y_pred_SVM = SVM_classifier.predict(X_test)
    y_pred_KernelSVM = KernelSVM_classifier.predict(X_test)
    y_pred_NB = NB_classifier.predict(X_test)
    y_pred_DecTree = DecTree_classifier.predict(X_test)
    y_pred_RanForest = RanForest_classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import accuracy_score
    Accuracy_Classifier[0,0] = 'KNN'
    Accuracy_Classifier[0,1] = 'LogReg'
    Accuracy_Classifier[0,2] = 'SVM'
    Accuracy_Classifier[0,3] = 'KernelSVM'
    Accuracy_Classifier[0,4] = 'NB'
    Accuracy_Classifier[0,5] = 'DecTree'
    Accuracy_Classifier[0,6] = 'RanForest'
    
    
    
    
    
#    Accuracy_Classifier['KNN']=accuracy_score(y_test, y_pred_KNN, normalize=True, sample_weight=None)
#    Accuracy_Classifier['LogReg']=accuracy_score(y_test, y_pred_LogReg, normalize=True, sample_weight=None)
#    Accuracy_Classifier['SVM']=accuracy_score(y_test, y_pred_SVM, normalize=True, sample_weight=None)
#    Accuracy_Classifier['KernelSVM']=accuracy_score(y_test, y_pred_KernelSVM, normalize=True, sample_weight=None)
#    Accuracy_Classifier['NB']=accuracy_score(y_test, y_pred_NB, normalize=True, sample_weight=None)
#    Accuracy_Classifier['DecTree']=accuracy_score(y_test, y_pred_DecTree, normalize=True, sample_weight=None)
#    Accuracy_Classifier['RanForest']=accuracy_score(y_test, y_pred_RanForest, normalize=True, sample_weight=None)

