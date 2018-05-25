# -*- coding: utf-8 -*-
"""
Created on Thu May 24 12:07:15 2018

@author: ANKIT
"""

# import modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from one_hot_encoding import one_hot_encoder
import time, os
import sys

# sys info
print(sys.version)

# load data
traindata = pd.read_csv('./dataset/train.csv')
testdata = pd.read_csv('./dataset/test.csv')

# ============================================ pre-process data ===============================================
numeric_cols = traindata._get_numeric_data().columns.tolist()

# Detection and Eliminatio of Outliers
for label in numeric_cols:
    traindata = traindata[np.abs(traindata[label]-traindata[label].mean()) <= (3.5*traindata[label].std())]   

# Taking care of missing values : Imputation
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
traindata[numeric_cols] = my_imputer.fit_transform(traindata[numeric_cols])

traindata.dropna(axis=1, inplace=True)
categorical_cols = []
for col in traindata:
    if(col not in numeric_cols):
        categorical_cols.append(col)
        print(traindata[col].value_counts())
        
# Taking care of missing values in Columns with categorical data where Imputation wont work
if(traindata.isnull().sum().sum()):
    traindata.dropna(inplace=True)
traindata.fillna(method='ffill', inplace=True)

   
y = traindata.iloc[:, -1].values
train_set = traindata.drop(['id', 'P'], axis=1)
numeric_cols.remove('id')
numeric_cols.remove('P')

# one-hot encoding
data = train_set[categorical_cols]
one_hot_encoded_data_train = one_hot_encoder(data, categorical_cols)
X = pd.concat([train_set[numeric_cols], one_hot_encoded_data_train], axis=1)

# Encoding the Independent Varialble
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder = LabelEncoder()
#catg_index = [0,3,4,5,6,8,9,11,12]
#for item in catg_index:
#    train_set[:, item] = labelencoder.fit_transform(train_set[:, item])
#onehotencoder = OneHotEncoder(categorical_features = catg_index)
#X = onehotencoder.fit_transform(train_set).toarray()

# ====================== Splitting the dataset into the Training set and Test set =============================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ============================================== Fitting Model ===============================================
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# ===================================================================================================
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# ==================================================================================================
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'poly', random_state = 0)
classifier.fit(X_train, y_train)

# ==================================================================================================
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# ==================================================================================================
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# ======================================== Predicting the Test set results ===================================
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# ================================================ Data Visualization =========================================

