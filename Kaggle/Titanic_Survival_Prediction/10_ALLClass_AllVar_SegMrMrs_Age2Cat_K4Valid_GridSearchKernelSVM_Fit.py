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
#import keras
#from keras.models import Sequential
#from keras.layers import Dense

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset_submission = pd.read_csv('test.csv')

# Extracting the Mr Mrs details for training set

datset_NamingAddress = pd.DataFrame.copy(dataset)
commaaddress = dotaddress = 0
for individuals in range(0,891):
    individualName = datset_NamingAddress.iloc[individuals,[3]].values[0]
    for findingAddress in range(0,len(individualName)):
        if str(individualName[findingAddress]) == ',':
            commaaddress = findingAddress
        elif str(individualName[findingAddress]) == '.':
            dotaddress = findingAddress
            continue
    title = individualName[commaaddress+2:dotaddress]
    
    if individualName[commaaddress+2:dotaddress] in ['Mr', 'Mrs', 'Miss', 'Master']:
        datset_NamingAddress.iloc[[individuals],[3]] = individualName[commaaddress+2:dotaddress]
    elif individualName[commaaddress+2:dotaddress] in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir']:
        datset_NamingAddress.iloc[[individuals],[3]] = 'Mr'
    elif individualName[commaaddress+2:dotaddress] in ['the Countess', 'Mme', 'Mrs. Martin (Elizabeth L', 'Lady']:
        datset_NamingAddress.iloc[[individuals],[3]] = 'Mrs'
    elif individualName[commaaddress+2:dotaddress] in ['Mlle', 'Ms']:
        datset_NamingAddress.iloc[[individuals],[3]] = 'Miss'
    elif str(individualName[commaaddress+2:dotaddress]) =='Dr':
        if datset_NamingAddress.iloc[[individuals],[4]].values =='Male':
            datset_NamingAddress.iloc[[individuals],[3]] = 'Mr'
        else:
            datset_NamingAddress.iloc[[individuals],[3]] = 'Mrs'
            
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
            
# Filling age based on Miss, Mr, Mrs, Master in training dataset

Mr_Age = []
Mrs_Age = []
Master_Age = []
Miss_Age = []

for AgeAverage in range(0,len(datset_NamingAddress)):
    individualAge = datset_NamingAddress.iloc[AgeAverage,[5]].values[0]
    if str(individualAge) != 'nan':
        if str(datset_NamingAddress.iloc[AgeAverage,[3]].values[0]) == 'Mr':
            Mr_Age.append(individualAge)
        elif str(datset_NamingAddress.iloc[AgeAverage,[3]].values[0]) == 'Mrs':
            Mrs_Age.append(individualAge)
        elif str(datset_NamingAddress.iloc[AgeAverage,[3]].values[0]) == 'Master':
            Master_Age.append(individualAge)
        elif str(datset_NamingAddress.iloc[AgeAverage,[3]].values[0]) == 'Miss':
            Miss_Age.append(individualAge)
Mr_mean = sum(Mr_Age)/len(Mr_Age)
Mrs_mean = sum(Mrs_Age)/len(Mrs_Age)
Master_mean = sum(Master_Age)/len(Master_Age)
Miss_mean = sum(Miss_Age)/len(Miss_Age)

for AgeAverageFill in range(0,len(datset_NamingAddress)):
    individualAgeFill = datset_NamingAddress.iloc[AgeAverageFill,[5]].values[0]
    if str(individualAgeFill) == 'nan':
        if str(datset_NamingAddress.iloc[AgeAverageFill,[3]].values[0]) == 'Mr':
            datset_NamingAddress.iloc[AgeAverageFill,[5]] = Mr_mean
        elif str(datset_NamingAddress.iloc[AgeAverageFill,[3]].values[0]) == 'Mrs':
            datset_NamingAddress.iloc[AgeAverageFill,[5]] = Mrs_mean
        elif str(datset_NamingAddress.iloc[AgeAverageFill,[3]].values[0]) == 'Master':
            datset_NamingAddress.iloc[AgeAverageFill,[5]] = Master_mean
        elif str(datset_NamingAddress.iloc[AgeAverageFill,[3]].values[0]) == 'Miss':
            datset_NamingAddress.iloc[AgeAverageFill,[5]] = Miss_mean

# Filling age based on Miss, Mr, Mrs, Master in test dataset

Mr_Age_s = []
Mrs_Age_s = []
Master_Age_s = []
Miss_Age_s = []

for AgeAverage_s in range(0,len(datset_submission_NamingAddress)):
    individualAge_s = datset_submission_NamingAddress.iloc[AgeAverage_s,[4]].values[0]
    if str(individualAge_s) != 'nan':
        if str(datset_submission_NamingAddress.iloc[AgeAverage_s,[2]].values[0]) == 'Mr':
            Mr_Age_s.append(individualAge_s)
        elif str(datset_submission_NamingAddress.iloc[AgeAverage_s,[2]].values[0]) == 'Mrs':
            Mrs_Age_s.append(individualAge_s)
        elif str(datset_submission_NamingAddress.iloc[AgeAverage_s,[2]].values[0]) == 'Master':
            Master_Age_s.append(individualAge_s)
        elif str(datset_submission_NamingAddress.iloc[AgeAverage_s,[2]].values[0]) == 'Miss':
            Miss_Age_s.append(individualAge_s)
Mr_mean_s = sum(Mr_Age_s)/len(Mr_Age_s)
Mrs_mean_s = sum(Mrs_Age_s)/len(Mrs_Age_s)
Master_mean_s = sum(Master_Age_s)/len(Master_Age_s)
Miss_mean_s = sum(Miss_Age_s)/len(Miss_Age_s)

for AgeAverageFill_s in range(0,len(datset_submission_NamingAddress)):
    individualAgeFill_s = datset_submission_NamingAddress.iloc[AgeAverageFill_s,[4]].values[0]
    if str(individualAgeFill_s) == 'nan':
        if str(datset_submission_NamingAddress.iloc[AgeAverageFill_s,[2]].values[0]) == 'Mr':
            datset_submission_NamingAddress.iloc[AgeAverageFill_s,[4]] = Mr_mean_s
        elif str(datset_submission_NamingAddress.iloc[AgeAverageFill_s,[2]].values[0]) == 'Mrs':
            datset_submission_NamingAddress.iloc[AgeAverageFill_s,[4]] = Mrs_mean_s
        elif str(datset_submission_NamingAddress.iloc[AgeAverageFill_s,[2]].values[0]) == 'Master':
            datset_submission_NamingAddress.iloc[AgeAverageFill_s,[4]] = Master_mean_s
        elif str(datset_submission_NamingAddress.iloc[AgeAverageFill_s,[2]].values[0]) == 'Miss':
            datset_submission_NamingAddress.iloc[AgeAverageFill_s,[4]] = Miss_mean_s

''''# Removing all the ageless rows
dataset_Ageless =pd.DataFrame.copy(datset_NamingAddress)
AgeLessRowList = []
for AgeCheck in range(0,891):
    a = dataset_Ageless.iloc[AgeCheck,[5]].values[0]
    if str(a) == 'nan':
        AgeLessRowList.append(AgeCheck)
        
dataset_Ageless = dataset_Ageless.drop(AgeLessRowList,axis = 0)'''

# Segregating X and y values
X = datset_NamingAddress.iloc[:, [2,3,4,5,6,7]].values
y = datset_NamingAddress.iloc[:,1]

X_submission = datset_submission_NamingAddress.iloc[:, [1,2,3,4,5,6]].values
index_list = []
 
'''# Segregating X and y values for AgeLess
X = dataset_Ageless.iloc[:, [2,3,4,5,6,7]].values
y = dataset_Ageless.iloc[:,1]
dataset_submission = pd.read_csv('test.csv')
X_submission = datset_submission_NamingAddress.iloc[:, [1,2,3,4,5,6]].values
index_list = []'''


# Creating a DF to save the accuracies for different classifiers
for number in range(1, 101):
    index_list.append(number)
Accuracy_Classifier=pd.DataFrame(index = index_list, columns=['KNN', 'LogReg', 'SVM', 'KernelSVM', 'NB', 'DecTree', 'RanForest','ANN'])

# Encoding categorical data
# Encoding the Independent Variable Making all strings to values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_Title = LabelEncoder()
X[:, 1] = labelencoder_X_Title.fit_transform(X[:, 1])

labelencoder_X_MF = LabelEncoder()
X[:, 2] = labelencoder_X_MF.fit_transform(X[:, 2])

labelencoder_X_submission_Title = LabelEncoder()
X_submission[:, 1] = labelencoder_X_submission_Title.fit_transform(X_submission[:, 1])

labelencoder_X_submission_MF = LabelEncoder()
X_submission[:, 2] = labelencoder_X_submission_MF.fit_transform(X_submission[:, 2])

'''# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)
imputer_submission = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_submission = imputer_submission.fit(X_submission)
X_submission = imputer_submission.transform(X_submission)'''

# Encoding categorical data
# Encoding the Independent Variable  - disecting the equall classes seperate

onehotencoder_X = OneHotEncoder(categorical_features = [0,1,2])
X = onehotencoder_X.fit_transform(X).toarray()

onehotencoder_X_submission = OneHotEncoder(categorical_features = [0,1,2])
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
    
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Classifier fit for KNN
    
KNN_classifier = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=2)
KNN_classifier.fit(X_train, y_train)

# Calculating Accuracy for the KNN Classifiers using K Fold
from sklearn.model_selection import cross_val_score
KNN_accuracies = cross_val_score(estimator = KNN_classifier,  X = X, y = y , cv = 10)
print('KNN :'+str(KNN_accuracies.mean())+','+str(KNN_accuracies.std()))

# Classifier fit for Log Reg
  
LogReg_classifier = LogisticRegression(random_state = 0)
LogReg_classifier.fit(X_train, y_train)

# Calculating Accuracy for the Log Reg Classifiers using K Fold

from sklearn.model_selection import cross_val_score
LogReg_accuracies = cross_val_score(estimator = LogReg_classifier,  X = X, y = y , cv = 10)
print('Logistic Regression :'+str(LogReg_accuracies.mean())+','+str(LogReg_accuracies.std()))

# Classifier fit for SVC
    
SVM_classifier = SVC(kernel = 'linear', random_state = 0)
SVM_classifier.fit(X_train, y_train)

# Calculating Accuracy for the SVC Classifiers using K Fold

from sklearn.model_selection import cross_val_score
SVM_accuracies = cross_val_score(estimator = SVM_classifier,  X = X, y = y , cv = 10)
print('SVM :'+str(SVM_accuracies.mean())+','+str(SVM_accuracies.std()))

# Classifier fit for Kernel SVC

KernelSVM_classifier = SVC(kernel = 'rbf')
KernelSVM_classifier.fit(X_train, y_train)

# Calculating Accuracy for the Kernel SVC Classifiers using K Fold

from sklearn.model_selection import cross_val_score
KernelSVM_accuracies = cross_val_score(estimator = KernelSVM_classifier,  X = X, y = y , cv = 10)
print('KernelSVM :'+str(KernelSVM_accuracies.mean())+','+str(KernelSVM_accuracies.std()))

# Applying Grid Search to find the best kernel SVM model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 2], 'kernel': ['rbf'], 'gamma': [0.17,0.21]}]

grid_search = GridSearchCV(estimator = KernelSVM_classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_accuracy, best_parameters)

# Classifier fit for NB
    
NB_classifier = GaussianNB()
NB_classifier.fit(X_train, y_train)

# Calculating Accuracy for the NB Classifiers using K Fold

from sklearn.model_selection import cross_val_score
NB_accuracies = cross_val_score(estimator = NB_classifier,  X = X, y = y , cv = 10)
print('Naive bayes :'+str(NB_accuracies.mean())+','+str(NB_accuracies.std()))

# Classifier fit for Decision Tree
    
DecTree_classifier = DecisionTreeClassifier(criterion = 'entropy')
DecTree_classifier.fit(X_train, y_train)

# Calculating Accuracy for the Decision Tree Classifiers using K Fold

from sklearn.model_selection import cross_val_score
DecTree_accuracies = cross_val_score(estimator = DecTree_classifier,  X = X, y = y , cv = 10)
print('Decision Tree :'+str(DecTree_accuracies.mean())+','+str(DecTree_accuracies.std()))

# Classifier fit for Random Forest
    
RanForest_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = None)
RanForest_classifier.fit(X_train, y_train)

# Calculating Accuracy for the Decision Tree Classifiers using K Fold

from sklearn.model_selection import cross_val_score
RanForest_accuracies = cross_val_score(estimator = RanForest_classifier, X = X, y = y , cv = 10)
print('Random Forest :'+str(RanForest_accuracies.mean())+','+str(RanForest_accuracies.std()))

# Fit and predict on Submission/Test Data
KernelSVM_classifier_Submission = SVC(kernel = 'rbf')
KernelSVM_classifier_Submission.fit(X, y)
    
y_pred_submission = KernelSVM_classifier_Submission.predict(X_submission)



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