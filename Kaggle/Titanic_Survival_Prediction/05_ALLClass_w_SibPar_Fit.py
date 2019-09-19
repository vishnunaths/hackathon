# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 21:41:19 2019

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 15:15:19 2019

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
#import keras
#from keras.models import Sequential
#from keras.layers import Dense

# Importing the dataset
dataset = pd.read_csv('train.csv')
#dataset_Ageless =pd.DataFrame(dataset)
#AgeLessRowList = []

'''# Removing all the ageless rows

for AgeCheck in range(0,891):
    a = dataset_Ageless.iloc[AgeCheck,[5]].values[0]
    if str(a) == 'nan':
        AgeLessRowList.append(AgeCheck)
        
dataset_Ageless = dataset_Ageless.drop(AgeLessRowList,axis = 0)'''

# Segregating X and y values
X = dataset.iloc[:, [2,4,5,6,7]].values
y = dataset.iloc[:,1]
dataset_submission = pd.read_csv('test.csv')
X_submission = dataset_submission.iloc[:, [1,3,4,5,6]].values
index_list = []
 
'''# Segregating X and y values for AgeLess
X = dataset_Ageless.iloc[:, [2,4,5]].values
y = dataset_Ageless.iloc[:,1]
dataset_submission = pd.read_csv('test.csv')
X_submission = dataset_submission.iloc[:, [1,3,4]].values
index_list = []'''

# Creating a DF to save the accuracies for different classifiers
for number in range(1, 101):
    index_list.append(number)
Accuracy_Classifier=pd.DataFrame(index = index_list, columns=['KNN', 'LogReg', 'SVM', 'KernelSVM', 'NB', 'DecTree', 'RanForest','ANN'])

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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    # Feature Scaling
    
    sc_X_train = StandardScaler()
    X_train = sc_X_train.fit_transform(X_train)

    sc_X_test = StandardScaler()
    X_test = sc_X_test.fit_transform(X_test)

    sc_X_submission = StandardScaler()
    X_submission = sc_X_submission.fit_transform(X_submission)
    
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)
    
    KNN_classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2)
    KNN_classifier.fit(X_train, y_train)
    
    LogReg_classifier = LogisticRegression(random_state = 0)
    LogReg_classifier.fit(X_train, y_train)
    
    SVM_classifier = SVC(kernel = 'linear', random_state = 0)
    SVM_classifier.fit(X_train, y_train)

    KernelSVM_classifier = SVC(kernel = 'rbf', random_state = 0)
    KernelSVM_classifier.fit(X_train, y_train)
    
    NB_classifier = GaussianNB()
    NB_classifier.fit(X_train, y_train)
    
    DecTree_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    DecTree_classifier.fit(X_train, y_train)
    
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
    
    # Fit and predict on Submission/Test Data
    KernelSVM_classifier_Submission = SVC(kernel = 'rbf', random_state = 0)
    KernelSVM_classifier_Submission.fit(X, y)
    
    y_pred_submission = KernelSVM_classifier_Submission.predict(X_submission)


    # Making the Confusion Matrix
    from sklearn.metrics import accuracy_score   
    Accuracy_Classifier.iloc[[NumberOfRuns],[0]]=accuracy_score(y_test, y_pred_KNN, normalize=True, sample_weight=None)
    Accuracy_Classifier.iloc[[NumberOfRuns],[1]]=accuracy_score(y_test, y_pred_LogReg, normalize=True, sample_weight=None)
    Accuracy_Classifier.iloc[[NumberOfRuns],[2]]=accuracy_score(y_test, y_pred_SVM, normalize=True, sample_weight=None)
    Accuracy_Classifier.iloc[[NumberOfRuns],[3]]=accuracy_score(y_test, y_pred_KernelSVM, normalize=True, sample_weight=None)
    Accuracy_Classifier.iloc[[NumberOfRuns],[4]]=accuracy_score(y_test, y_pred_NB, normalize=True, sample_weight=None)
    Accuracy_Classifier.iloc[[NumberOfRuns],[5]]=accuracy_score(y_test, y_pred_DecTree, normalize=True, sample_weight=None)
    Accuracy_Classifier.iloc[[NumberOfRuns],[6]]=accuracy_score(y_test, y_pred_RanForest, normalize=True, sample_weight=None)
    Accuracy_Classifier.iloc[[NumberOfRuns],[7]]=0#accuracy_score(y_test, y_pred_ANN, normalize=True, sample_weight=None)

Accuracy_Classifier.mean(axis=0)

'''    # Fitting Classifier to the Training set
    # Initialising the ANN
    ANN_classifier = Sequential()

    # Adding the input layer and the first hidden layer
    ANN_classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu', input_dim = 6))

    # Adding the second hidden layer
    ANN_classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu'))

    # Adding the output layer
    ANN_classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

    # Compiling the ANN
    ANN_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting the ANN to the Training set
    ANN_classifier.fit(X, y, batch_size = 10, nb_epoch = 100)'''
    
'''    y_pred_ANN = ANN_classifier.predict(X_test)
    for Prob in range(0, len(y_pred_ANN)):
        if y_pred_ANN[Prob] > 0.5:
            y_pred_ANN[Prob] = 1
        else:
            y_pred_ANN[Prob] = 0
    y_pred_submission = ANN_classifier.predict(X_submission)
    for Prob in range(0, len(y_pred_submission)):
        if y_pred_submission[Prob] > 0.5:
            y_pred_submission[Prob] = 1
        else:
            y_pred_submission[Prob] = 0'''