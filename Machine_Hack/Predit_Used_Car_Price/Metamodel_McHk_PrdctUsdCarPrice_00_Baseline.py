# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:58:49 2019

@author: kzt9qh
"""

# Timer start
import timeit
start = timeit.default_timer()
print("$$time count started$$")

#Dataset Import
import pandas as pd
dataset = pd.read_csv('Data_Train_test.csv')
print("$$dataset imported$$")
    
# Data preprocessing
# Function for Data preprocesing
def dataEncoding(datasetName,colDetail,isNumeric=0,LabelEnc_req=1,OneHot_req=1):
    """Function to convert column into machine learning format
    variable = isNumeric, LabelEnc_req,OneHot_req"""
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    if isNumeric == 0:
        X = datasetName.iloc[:,colDetail]
        X = X.values.reshape((len(X),1))
        if LabelEnc_req == 1 and OneHot_req == 1:
            LabEnc_X = LabelEncoder()
            LabEnc_X = LabEnc_X.fit(X)
            X = LabEnc_X.transform(X) 
            X = X.reshape((len(X),1))
            onehotEnc_X = OneHotEncoder(categorical_features = [0])
            onehotEnc_X = onehotEnc_X.fit(X)
            X = onehotEnc_X.transform(X).toarray()
            X = pd.DataFrame(X)
            X = X.values
        else:
            LabEnc_X = LabelEncoder()
            LabEnc_X = LabEnc_X.fit(X)
            X = LabEnc_X.transform(X) 
            X = X.reshape((len(X),1))
    else:
        X = datasetName.iloc[:,colDetail]
        X = X.values.reshape((len(X),1))
        sc_X = StandardScaler()
        X = sc_X.fit_transform(X)
    return X

X1 = dataEncoding(dataset,0)
X2 = dataEncoding(dataset,1)
dataset.iloc[:,2] = dataset.iloc[:,2].fillna('Not Provided')
dataset.iloc[:,3] = dataset.iloc[:,3].fillna('Not Provided')
dataset.iloc[:,4] = dataset.iloc[:,4].fillna('Not Provided')
X3 = dataEncoding(dataset,2)
X4 = dataEncoding(dataset,3)
X5 = dataEncoding(dataset,4)
X6 = dataEncoding(dataset,8)
#Converting Manufac. Year to Car Age
dataset.iloc[:,9] = 2019 - dataset.iloc[:,9]
X7 = dataEncoding(dataset,9,isNumeric=1)
X8 = dataEncoding(dataset,10,isNumeric=1)
X9 = dataEncoding(dataset,11)
X10 = dataEncoding(dataset,12)
for ownerHand in range(len(dataset)):
    if dataset.iloc[ownerHand,13] == 'First':
        dataset.iloc[ownerHand,13] = 4
    elif dataset.iloc[ownerHand,13] == 'Second':
        dataset.iloc[ownerHand,13] = 3
    elif dataset.iloc[ownerHand,13] == 'Third':
        dataset.iloc[ownerHand,13] = 2
    else:
        dataset.iloc[ownerHand,13] = 1
X11 = dataEncoding(dataset,13,isNumeric=1)
dataset.iloc[:,14] = dataset.iloc[:,14].fillna(18.14)
X12 = dataEncoding(dataset,14,isNumeric=1)
dataset.iloc[:,16] = dataset.iloc[:,16].fillna(1616.1616)
X13 = dataEncoding(dataset,16,isNumeric=1)    
dataset.iloc[:,18] = dataset.iloc[:,18].fillna(112)
X14 = dataEncoding(dataset,18,isNumeric=1)
dataset.iloc[:,20] = dataset.iloc[:,20].fillna(5)
X15 = dataEncoding(dataset,20,isNumeric=1)
for newPrice in range(len(dataset)):
    if (dataset.iloc[newPrice,23] == 0) or (dataset.iloc[newPrice,23] == 'NaN'):
        dataset.iloc[newPrice,23] = 'Not Provided'
    elif 0 < dataset.iloc[newPrice,23] < 500000:
        dataset.iloc[newPrice,23] = 'Less than 5 lakh'
    elif 500000 <= dataset.iloc[newPrice,23] < 1000000:
        dataset.iloc[newPrice,23] = 'between 5 to 10 lakh'
    elif 1000000 <= dataset.iloc[newPrice,23] < 1500000:
        dataset.iloc[newPrice,23] = 'between 10 to 15 lakh'
    elif 1500000 <= dataset.iloc[newPrice,23] < 2000000:
        dataset.iloc[newPrice,23] = 'between 15 to 20 lakh'
    elif 2000000 <= dataset.iloc[newPrice,23] < 3000000:
        dataset.iloc[newPrice,23] = 'between 20 to 30 lakh'
    elif 3000000 <= dataset.iloc[newPrice,23] < 5000000:
        dataset.iloc[newPrice,23] = 'between 30 to 50 lakh'
    elif 5000000 <= dataset.iloc[newPrice,23] < 7500000:
        dataset.iloc[newPrice,23] = 'between 50 to 75 lakh'
    elif 7500000 <= dataset.iloc[newPrice,23] < 10000000:
        dataset.iloc[newPrice,23] = 'between 75 to 1 crore'
    else:
        dataset.iloc[newPrice,23] = 'more than 1 crore'
X16 = dataEncoding(dataset,23)
y = dataset.iloc[:6019,24]
y = y.values

#Training and test set creation
import numpy as np
X_train_test = np.concatenate((X1,X2,X3,X4,X6,X7,X8,X10,X11,X12,X13,X14,X15),axis=1)
X = X_train_test[:6019,:]
X_submit = X_train_test[6019:,:]   

# removing the na Values (43)
removeRows43 = []
for l2 in range(6019):
    if dataset.iloc[l2,16] == 1616.1616:
        removeRows43.append(l2)

# Deleting missinf values
X = np.delete(X, removeRows43, axis=0)
y = np.delete(y, removeRows43, axis=0)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
print('$$Splitting X&y-success$$')

# Fitting the Random Forest Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
RF_regressor = RandomForestRegressor(n_estimators=3000,min_samples_leaf=1,
                                     min_samples_split=3,max_depth=16)
RF_regressor.fit(X_train,y_train)

# Predicting a new result
y_pred = RF_regressor.predict(X_test)

print('RF Regressor')
from sklearn.metrics import r2_score,mean_absolute_error
r2_score = r2_score(y_test, y_pred, sample_weight=None)
print(r2_score)

mean_absolute_error = mean_absolute_error(y_test, y_pred, sample_weight=None)
print(mean_absolute_error)

'''# Applying Grid Search to find the best kernel SVM model and the best parameters
from sklearn.model_selection import GridSearchCV
min_samples_splits = range(2,5,1)
parameters = [{'n_estimators': [3000],'min_samples_leaf':[1],'max_depth':[16],
               'min_samples_split':[3]}]

grid_search = GridSearchCV(estimator = RandomForestRegressor(),
                           param_grid = parameters,
                           scoring = 'r2',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_accuracy, best_parameters)'''


'''# predicting submit values
from sklearn.ensemble import RandomForestRegressor
RF_regressor = RandomForestRegressor(n_estimators=3000,min_samples_leaf=1,
                                     min_samples_split=3,max_depth=16)
RF_regressor.fit(X,y)

y_submit = RF_regressor.predict(X_submit)'''







stop = timeit.default_timer()
print('Time :',(stop - start)//3600,'Hrs',((stop - start)%3600)//60,'Mins',(int((stop - start)%3600)%60),'secs')

