# -*- coding: utf-8 -*-
"""
Importing the Libraries
"""

import pandas as pd 
import numpy as np                     # For mathematical calculations 
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

"""Importing Files"""

train=pd.read_csv("Train.csv")
test=pd.read_csv("Test.csv")

train_dataset = pd.concat([train,test],axis =0, sort=False)

"""Data Preprocessing"""

X_Vendor_Code = train_dataset['Vendor_Code']
X_Vendor_Code  = X_Vendor_Code.values.reshape((len(X_Vendor_Code),1))
labelencoder_X_Vendor_Code = LabelEncoder()
labelencoder_X_Vendor_Code = labelencoder_X_Vendor_Code.fit(X_Vendor_Code)
X_Vendor_Code = labelencoder_X_Vendor_Code.transform(X_Vendor_Code)
X_Vendor_Code = X_Vendor_Code.reshape((len(X_Vendor_Code),1))
onehotencoder_X_Vendor_Code = OneHotEncoder()
onehotencoder_X_Vendor_Code = onehotencoder_X_Vendor_Code.fit(X_Vendor_Code)
X_Vendor_Code = onehotencoder_X_Vendor_Code.transform(X_Vendor_Code).toarray()

X_GL_Code = train_dataset['GL_Code']
X_GL_Code  = X_GL_Code.values.reshape((len(X_GL_Code),1))
labelencoder_X_GL_Code = LabelEncoder()
labelencoder_X_GL_Code = labelencoder_X_GL_Code.fit(X_GL_Code)
X_GL_Code = labelencoder_X_GL_Code.transform(X_GL_Code)
X_GL_Code = X_GL_Code.reshape((len(X_GL_Code),1))
onehotencoder_X_GL_Code = OneHotEncoder()
onehotencoder_X_GL_Code = onehotencoder_X_GL_Code.fit(X_GL_Code)
X_GL_Code = onehotencoder_X_GL_Code.transform(X_GL_Code).toarray()

X_inv_id = train_dataset['Inv_Id']
X_inv_id = pd.DataFrame(X_inv_id)
X_inv_id = X_inv_id.values

y = train_dataset['Product_Category']
y = pd.DataFrame(y)
y = y.values

"""re and NLTK"""

class CleanText(BaseEstimator, TransformerMixin):
        def remove_mentions(self, input_text):
            return re.sub(r'@\w+', '', input_text)
    
        def remove_punctuation(self, input_text):
            # Make translation table
            punct = string.punctuation
            trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
            return input_text.translate(trantab)
    
        def remove_digits(self, input_text):
            return re.sub('\d+', '', input_text)
    
        def to_lower(self, input_text):
            return input_text.lower()
    
        def remove_stopwords(self, input_text):
            stopwords_list = stopwords.words('english')
            # Some words which might indicate a certain sentiment are kept via a whitelist
            words = input_text.split() 
            clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 1] 
            return " ".join(clean_words) 
    
        def Lemmatizing(self, input_text):
            lemmatizer = WordNetLemmatizer()
            words = input_text.split() 
            stemmed_words = [lemmatizer.lemmatize(word) for word in words]
            return " ".join(stemmed_words)
    
        def fit(self, X, y=None, **fit_params):
            return self
    
        def transform(self, X, **transform_params):
            clean_X = X.apply(self.remove_mentions).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.Lemmatizing)
            return clean_X

ct = CleanText()
train_dataset['Item_Description'] = ct.fit_transform(train_dataset.Item_Description)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english') 
tfidf = tfidf_vectorizer.fit_transform(train_dataset['Item_Description']).toarray()

X_train_test = np.concatenate((X_Vendor_Code,X_GL_Code,X_inv_id,tfidf),axis=1)
X = X_train_test[:5566,:]
y = y[:5566]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

""" pREDICTING tEST results """
y_pred = classifier.predict(X_test)
test = X_train_test[5566:,:]
test_pred = classifier.predict(test)
np.savetxt('submission.csv', test_pred, fmt='%s')