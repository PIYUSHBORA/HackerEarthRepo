# -*- coding: utf-8 -*-
"""
Created on Thu May 24 15:15:54 2018

@author: ANKIT
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from one_hot_encoding import one_hot_encoder
from main import traindata, classifier, one_hot_encoded_data_train
import csv
import sys

# sys info
print(sys.version)

# load data
testdata = pd.read_csv('./dataset/test.csv')
temp = traindata.columns.tolist()
temp.remove('P')
testdata = testdata[temp]
perid = testdata.iloc[:, 0:1].values

# ============================================ pre-process data ===============================================
numeric_cols = testdata._get_numeric_data().columns.tolist()

# Taking care of missing values : Imputation
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
testdata[numeric_cols] = my_imputer.fit_transform(testdata[numeric_cols])

categorical_cols = []
for col in testdata:
    if(col not in numeric_cols):
        categorical_cols.append(col)
        print(testdata[col].value_counts())
        
# Taking care of missing values in Columns with categorical data where Imputation wont work
testdata.fillna(method='ffill', inplace=True)

test_set = testdata.drop(['id'], axis=1)
numeric_cols.remove('id')

# one-hot encoding
data = test_set[categorical_cols]
one_hot_encoded_data_test = one_hot_encoder(data, categorical_cols)
feature_difference = set(one_hot_encoded_data_train) - set(one_hot_encoded_data_test)
feature_difference_df = pd.DataFrame(data=np.zeros((one_hot_encoded_data_test.shape[0], 
                                                    len(feature_difference))),
    columns=list(feature_difference))
one_hot_encoded_data_test = one_hot_encoded_data_test.join(feature_difference_df)
X = pd.concat([test_set[numeric_cols], one_hot_encoded_data_test], axis=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# ========================================= Predicting the Test set results ======================================
y_pred = classifier.predict(X)

# ========================================== Saving prediction to CSV file ==========================================
myFields = ['id', 'P']
i=0 
myFile = open('submission1.csv', 'w')
with myFile:
    writer = csv.DictWriter(myFile, fieldnames=myFields)  
    writer.writeheader()
    for i in range (len(perid)):
        if(y_pred[i] == 0):     writer.writerow({'id' : perid[i][0], 'P': '0'})
        else:                   writer.writerow({'id' : perid[i][0], 'P': '1'}) 
     
print("Writing complete")