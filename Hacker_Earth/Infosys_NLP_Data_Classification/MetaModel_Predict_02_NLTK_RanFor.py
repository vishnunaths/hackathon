# -*- coding: utf-8 -*-
"""HackerEarth_InfyEdgeVerve_NLP_Classification_00.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lff5kUbywy55W9gTRl3mdzUQiLEJtPk2

Hacker Earth Infy Edge Verve NLP Classification

Importing the Libraries
"""

import keras
import nltk
import pandas as pd
import numpy as np
import re
import codecs
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error

"""Importing Files"""

train_dataset = pd.read_csv("Train.csv")
test_dataset = pd.read_csv("Test.csv")
train_dataset = pd.concat([train_dataset,test_dataset],axis =0, sort=False,ignore_index=True)

train_dataset.head()

"""Data Preprocessing"""

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(train_dataset)):
    review = re.sub('[^a-zA-Z]', ' ', str(train_dataset['Item_Description'][i]))
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

"""Creating the Bag of Words model"""

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3018)
X4 = cv.fit_transform(corpus).toarray()

X1 = train_dataset.iloc[:,1]
X1  = X1.values.reshape((len(X1),1))
onehotencoder_X1 = OneHotEncoder()
onehotencoder_X1 = onehotencoder_X1.fit(X1)
X1 = onehotencoder_X1.transform(X1).toarray()

X2 = train_dataset.iloc[:,2]
X2  = X2.values.reshape((len(X2),1))
onehotencoder_X2 = OneHotEncoder()
onehotencoder_X2 = onehotencoder_X2.fit(X2)
X2 = onehotencoder_X2.transform(X2).toarray()

X3 = train_dataset.iloc[:,3]
X3  = X3.values.reshape((len(X2),1))

y = train_dataset.iloc[:5566,-1]
y  = y.values.reshape((len(y),1))
labelencoder_y = LabelEncoder()
labelencoder_y = labelencoder_y.fit(y)
y = labelencoder_y.transform(y)

np.unique(y)

X_train_test = np.concatenate((X1,X2,X3,X4),axis=1)
X = X_train_test[:5566,:]

"""Train Test Split"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35)

"""Prediction Using All Classifier

KNN Classifier
"""

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score    
Accuracy=accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(Accuracy*100,'%')

"""Logistic Regression"""

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score    
Accuracy=accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(Accuracy*100,'%')

"""Support Vector Machine"""

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score    
Accuracy=accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(Accuracy*100,'%')

"""Naive Bayes"""

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score    
Accuracy=accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(Accuracy*100,'%')

"""Decision Tree"""

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score    
Accuracy=accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(Accuracy*100,'%')

"""Random Forest"""

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 25, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score    
Accuracy=accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print(Accuracy*100,'%')

"""Training full data"""

X_submit = X_train_test[5566:,:]
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 25, criterion = 'entropy')
classifier.fit(X, y)
y_submit = classifier.predict(X_submit)

y_pred = labelencoder_y.inverse_transform(y_submit)

np.savetxt("submission.csv", y_pred, delimiter=",",fmt='%s')