# -*- coding: utf-8 -*-
"""
Created on Fri May 18 02:21:28 2018

@author: Guido Muscioni
"""

import pandas as pd
import glob
import numpy as np
import sklearn.model_selection as ms
#import sklearn.metrics as mt
import xgboost as xgb
import time
import csv
import geopy.distance
from geopy.distance import vincenty
from sklearn import preprocessing
from networkx import nx
import itertools
from threading import Thread
from multiprocessing import Pool as ThreadPool
import sklearn.metrics as mt
from random import randint
import time
#from multiprocessing.pool import ThreadPool

from math import sin, cos, sqrt, atan2, radians, inf
from scipy.spatial.distance import pdist, squareform

#%% Functions

def evaluation(y_test, y_pred):
    accuracy = mt.accuracy_score(y_test,y_pred)
    print("Accuracy: ", accuracy)
    f_score = mt.f1_score(y_test,y_pred, average = "weighted")
    print("F1 score: ", f_score)
    confusion_matrix = mt.confusion_matrix(y_test,y_pred)
    print("Confusion matrix: \n", confusion_matrix)
    return accuracy, f_score, confusion_matrix

#%% Main

train = pd.read_csv('../data/group/day4.csv', sep = ' ', header =None)
test = pd.read_csv('../data/group/day3.csv', sep =' ', header =None)
train_label = train.pop(train.columns[len(train.columns)-1])
test_label = test.pop(test.columns[len(test.columns)-1])


print('Training xgb number ' + str(1))             
# Training model            
dtrain = xgb.DMatrix(train, label=train_label)
dtest = xgb.DMatrix(test, label=test_label)
evallist  = [(dtest,'eval'), (dtrain,'train')]
param = {}
param['objective'] = 'multi:softmax'
param['eta'] = 0.2
param['max_depth'] = 100
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = len(np.unique(test_label)) + 2
num_round = 51
model = xgb.train( param, dtrain, num_round, evallist )
y_pred = model.predict(dtest)
accuracy, f_score, confusion_matrix = evaluation(y_pred, test_label)


