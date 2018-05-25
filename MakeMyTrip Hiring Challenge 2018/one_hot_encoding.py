# -*- coding: utf-8 -*-
"""
Created on Thu May 24 12:33:17 2018

@author: ANKIT
"""

import pandas as pd


def one_hot_encoder(data, categorical_cols):
    print("====================[Data Types]====================")
    print(data.dtypes)
    categorical_variables = categorical_cols
    data_one_hot_encoded = pd.get_dummies(data, columns=categorical_variables)
    return data_one_hot_encoded