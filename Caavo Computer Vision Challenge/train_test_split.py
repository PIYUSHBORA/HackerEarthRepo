# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 16:36:58 2018

@author: ANKIT
"""

import os
from sklearn.model_selection import train_test_split

os.chdir('./dataset/train')
path = os.getcwd()
directory = os.listdir()

for folder in directory:
    files_X = files_y = os.listdir(folder)
    X_train, X_test = train_test_split(files_X, test_size=0.25, random_state=0)
    
    for x in X_train:
        os.rename(os.path.join(path, folder, x) , os.path.join(path, '..\\new_train', folder, x))
        
    for x in X_test:
        os.rename(os.path.join(path, folder, x), os.path.join(path, '..\\new_validate', folder, x))