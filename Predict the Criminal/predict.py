# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 20:34:14 2018

@author: ANKIT
"""

# import modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
from one_hot_encoding import one_hot_encoder
import feature_index
import csv

testdata = pd.read_csv('./data/criminal_test.csv')
perid = testdata.iloc[:, 0:1].values
testdata = testdata.drop(['PERID'], axis=1)
data = testdata[feature_index.categorical]
data_one_hot_encoded = one_hot_encoder(data)
X = pd.concat([testdata, data_one_hot_encoded], axis=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# Load model
model = load_model('./model/checkpoints/criminal-ann-009-loss0.098-acc0.958.hdf5')

scores = model.predict(np.array(X))

myData = [["PERID", "Criminal"]]
i=0
for i in range (len(perid)):
    if(scores[i] > 0):
        score = 1
    else:
        score = 0
    myData.append([perid[i][0], score])
 
myFile = open('submission.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(myData)
     
print("Writing complete")