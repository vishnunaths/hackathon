# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:04:47 2019

@author: Naveen Subramanian
"""
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
   
start_Time = time.time()

workPath = 'V:/2019/WIP/Edelweiss_HackerEarth_I_II/WIP/PartI'

X = pd.read_csv(os.path.join(workPath, "train_LMS_CalculatedDataFile.csv"))
y = pd.read_csv(os.path.join(workPath, "train_foreclosure.csv"))

y = y['FORECLOSURE']    

y.fillna(-1, inplace = True)

X_RealTest = pd.read_csv("test_LMS_CalculatedDataFile.csv")

labelencoder = LabelEncoder()

X['PRODUCT'] = labelencoder.fit_transform(X['PRODUCT'])    
    
X['CITY'].fillna('NOCITY', inplace = True)
X['CITY'] = labelencoder.fit_transform(X['CITY'])

X_RealTest['PRODUCT'] = labelencoder.fit_transform(X_RealTest['PRODUCT'])

X_RealTest['CITY'].fillna('NOCITY', inplace = True)
X_RealTest['CITY'] = labelencoder.fit_transform(X_RealTest['CITY'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

y_test = pd.DataFrame(y_test)

y_test.to_csv('y_test.csv')

#KNN_classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2)
#KNN_classifier.fit(X_train, y_train)

#DecTree_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#DecTree_classifier.fit(X, y)

#DecTree_Regressor = DecisionTreeRegressor()
#DecTree_Regressor.fit(X, y)

RanFor_regressor = RandomForestRegressor(n_estimators=300)
RanFor_regressor.fit(X, y)

yPred = RanFor_regressor.predict(X_test);   

yPredDf = pd.DataFrame(yPred, columns=["FORECLOSURE"])

yPredDf.to_csv("yPred.csv")

y_test = y_test.values.tolist()

pos = 0;
neg = 0;
for i in range(0, len(y_test)):
    if yPred[i] == y_test[i]:
        pos +=1;
        
accuracy = (pos/ len(y_test)) * 100
print("Accuracy: $", accuracy)

y_PredReal = RanFor_regressor.predict(X_RealTest);

y_PredRealDf = pd.DataFrame(y_PredReal, columns=["FORECLOSURE"])

y_PredRealDf.to_csv("yPredReal.csv")

print("Total time" + str(time.time() - start_Time))