# -*- coding: utf-8 -*-
"""
Author: Ankit Dutta
"""
import feature_index
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


def encode_neural_net_y(y):
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    new_y = np_utils.to_categorical(encoded_y)
    return new_y


def one_hot_encoder(data):
    print("====================[Data Types]====================")
    print(data.dtypes)
    categorical_variables = feature_index.categorical
    data_one_hot_encoded = pd.get_dummies(data, columns=categorical_variables)
    return data_one_hot_encoded
