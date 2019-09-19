# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 07:24:50 2019

@author: SJMB
"""

# Importing

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
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

start = timeit.default_timer()

train_dataset = pd.read_csv('train.csv')
y = train_dataset.iloc[:,1].values
Column_Names = train_dataset.columns
train_dataset = train_dataset.iloc[:,-1]
test_dataset = pd.read_csv('test.csv')
test_dataset = test_dataset.iloc[:,-1]
train_dataset = pd.concat([train_dataset,test_dataset],axis =0,ignore_index=True)
print('$$Dataset imported-success$$')


X_ini = train_dataset.copy()
X_ini = pd.DataFrame(X_ini)
print("$$X,y Created$$")

newList = []
for xtok in range(len(X_ini)):
    sepWords = X_ini.iloc[xtok,0]
    sepWords = sepWords.split()
    for eachTok in sepWords:
        newList.append(int(eachTok))
maxValue = max(newList)
minValue = min(newList)
import collections
counter=collections.Counter(newList)
print(counter)
ohvIndex = []
ohvIndex_s = []
for ind in newList:
    if ind not in ohvIndex:
        ohvIndex.append(ind)
        ohvIndex_s.append(str(ind))

X_train_test = np.zeros((len(X_ini),8602))
X_train_test = pd.DataFrame(X_train_test,columns = ohvIndex_s)

for xAssign in range(len(X_ini)):
    asgWords = X_ini.iloc[xAssign,0]
    asgWords = asgWords.split()
    for eachAssign in asgWords:
        xIndex = ohvIndex_s.index(eachAssign)
        X_train_test.iloc[xAssign,xIndex] += 1
#    print('still to go : '+str(4824-xAssign))
print('$$Counting - success$$')
words_in_both_s = ['5573', '1177']    
X_train_test = X_train_test.drop(words_in_both_s, axis=1) # removed 600+
X = X_train_test.iloc[:3464,:]
y = y[:3464]
X_s = X_train_test.iloc[3464:,:]
print("$$X Concat done$$")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
print('$$Splitting X&y-success$$')

'''# Classifier
# Classifier fit for KNN
KNN_classifier = KNeighborsClassifier(n_neighbors=2, weights='distance', metric='minkowski', p=2)
KNN_classifier.fit(X_train,y_train)
    
y_pred = KNN_classifier.predict(X_test)
    
print('KNN')   
Accuracy_RanForest =accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(Accuracy_RanForest)
    
cnf_mat = confusion_matrix(y_test, y_pred, labels=[0,1], sample_weight=None)
print(cnf_mat)

# Classifier fit for Log Reg
  
LogReg_classifier = LogisticRegression()
LogReg_classifier.fit(X_train,y_train)

y_pred = LogReg_classifier.predict(X_test)

print('Logistic Regressor')   
Accuracy_RanForest =accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(Accuracy_RanForest)

cnf_mat = confusion_matrix(y_test, y_pred, labels=[0,1], sample_weight=None)
print(cnf_mat)'''

# Classifier fit for SVC
    
SVM_classifier = SVC(C=3,kernel='linear')
SVM_classifier.fit(X_train, y_train)

y_pred = SVM_classifier.predict(X_test)

print('SVM CLassifier')   
Accuracy_RanForest =accuracy_score(y_pred, y_test, normalize=True, sample_weight=None)
print(Accuracy_RanForest)

cnf_mat = confusion_matrix(y_test, y_pred, labels=[0,1], sample_weight=None)
print(cnf_mat)

'''# Classifier fit for Kernel SVC

KernelSVM_classifier = SVC(kernel = 'rbf')
KernelSVM_classifier.fit(X_train, y_train)

y_pred = KernelSVM_classifier.predict(X_test)
print('Kernel SVM')
   
Accuracy_RanForest =accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(Accuracy_RanForest)

cnf_mat = confusion_matrix(y_test, y_pred, labels=[0,1], sample_weight=None)
print(cnf_mat)

# Classifier fit for NB
    
NB_classifier = GaussianNB()
NB_classifier.fit(X_train, y_train)

y_pred = NB_classifier.predict(X_test)
print('NB')  
 
Accuracy_RanForest =accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(Accuracy_RanForest)

cnf_mat = confusion_matrix(y_test, y_pred, labels=[0,1], sample_weight=None)
print(cnf_mat)

# Classifier fit for Decision Tree
    
DecTree_classifier = DecisionTreeClassifier(criterion = 'entropy')
DecTree_classifier.fit(X_train, y_train)

y_pred = DecTree_classifier.predict(X_test)
print('DecTree_classifier')  
 
Accuracy_RanForest =accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(Accuracy_RanForest)

cnf_mat = confusion_matrix(y_test, y_pred, labels=[0,1], sample_weight=None)
print(cnf_mat)
# Classifier fit for Random Forest
    
RanForest_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = None)
RanForest_classifier.fit(X_train, y_train)

y_pred = RanForest_classifier.predict(X_test)
print('RanForest_classifier')  
 
Accuracy_RanForest =accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(Accuracy_RanForest)

cnf_mat = confusion_matrix(y_test, y_pred, labels=[0,1], sample_weight=None)
print(cnf_mat)

# Classifier fit for ANN
    
ANN_classifier = MLPClassifier(hidden_layer_sizes=(20,), activation='relu', solver='adam', alpha=0.0001, batch_size=100, learning_rate='invscaling', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.00001, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
ANN_classifier.fit(X_train, y_train)

y_pred = ANN_classifier.predict(X_test)
print('ANN')  
 
Accuracy_ANN =accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(Accuracy_ANN)

cnf_mat = confusion_matrix(y_test, y_pred, labels=[0,1], sample_weight=None)
print(cnf_mat)
'''
# submission prediction

classifier_s = SVC(kernel = 'linear',verbose=True)
classifier_s.fit(X, y)

y_pred_s = classifier_s.predict(X_s)

np.sum(y_pred_s)
stop = timeit.default_timer()
print('Time: ', stop - start)

