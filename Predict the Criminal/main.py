---# -*- coding: utf-8 -*-
"""
Author: Ankit Dutta
"""

# import modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from one_hot_encoding import one_hot_encoder
import feature_index
import time, os
import sys

# sys info
print(sys.version)

# load data
traindata = pd.read_csv('./data/criminal_train.csv')
testdata = pd.read_csv('./data/criminal_test.csv')

# Detection and Eliminatio of Outliers
labels = list(traindata)
for label in labels:
    traindata = traindata[np.abs(traindata[label]-traindata[label].mean()) <= (3.8*traindata[label].std())]

# pre-process data
traindata.isnull().sum()
num_cols = traindata._get_numeric_data().columns
for col in traindata:
    print(traindata[col].value_counts())

y = traindata.iloc[:, -1].values
traindata = traindata.drop(['PERID', 'Criminal'], axis=1)
# Except VESTR and ANALWT_C all others are Categorical data. Hence they need feature scaling.
# one-hot encoding
data = traindata[feature_index.categorical]
one_hot_encoded_data = one_hot_encoder(data)

# Splitting the dataset into the Training set and Test set
X = pd.concat([traindata, one_hot_encoded_data], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ========================================================Neural Network model==================================================================
# Initialising the ANN
model = Sequential()
# Adding the input layer and the first hidden layer
model.add(Dense(units = 70, kernel_initializer = 'glorot_uniform', use_bias=True, activation = 'relu', input_dim = 70))
# Adding the second hidden layer
model.add(Dense(units = 560, kernel_initializer = 'glorot_uniform', activation = 'relu'))
model.add(Dropout(0.5))
# Adding the second hidden layer
model.add(Dense(units = 280, kernel_initializer = 'glorot_uniform', activation = 'relu'))
model.add(Dropout(0.4))
# Adding the third hidden layer
model.add(Dense(units = 140, kernel_initializer = 'glorot_uniform', activation = 'relu'))
model.add(Dropout(0.3))
# Adding the fourth hidden layer
model.add(Dense(units = 70, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0.25))
# Adding the output layer
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
sgd = SGD(lr=0.1, momentum=0.8, decay=0.0, nesterov=False)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Helper: Save the model.
checkpointer = ModelCheckpoint(filepath="model/checkpoints/criminal-ann-{epoch:03d}-loss{val_loss:.3f}-acc{val_acc:.3f}.hdf5", verbose=1, save_best_only=True)
# Helper: TensorBoard
tb = TensorBoard(log_dir=os.path.join('model', 'logs', 'criminal-ann'))

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(monitor='val_loss', patience= 10, mode = 'auto')

# Helper: Save results.
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('model', 'logs', 'criminal-ann' + '-' + 'training-' + \
                                    str(timestamp) + '.log'))

# Fitting the ANN to the Training set
hist = model.fit(X_train, y_train, batch_size = 256, epochs = 100, 
          validation_data = (X_test, y_test), 
          callbacks=[tb, csv_logger, early_stopper, checkpointer])

plt.plot(hist.history["acc"], label='acc')
plt.plot(hist.history["val_acc"], label='val_acc')
plt.legend()
plt.savefig('performance')
plt.show()

scores = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
print("====================[TEST SCORE]====================")
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# Save model
model.save('neural_net_model.h5')
