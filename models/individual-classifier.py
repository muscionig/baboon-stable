# -*- coding: utf-8 -*-
"""
Created on Wed May 16 20:49:44 2018

@author: Guido Muscioni

Description: Individual classifier for the baboon dataset, 
Simple aggregation ove time
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

#%% Globals

global resultDict
resultDict = {}

global resultDataframe
resultDataframe = pd.DataFrame()


#%% Functions

current_milli_time = lambda: int(round(time.time() * 1000))

def readDataset(fileList):
    df = pd.DataFrame()
    for file in fileList:
        print('Reading: ' + file)
        temp = pd.read_csv(file, sep = ',')
        temp = temp.fillna(method = 'ffill')
        temp = temp.dropna()
        if(df.empty):
            df = temp.copy()
        else:
            df = df.append(temp.copy(), ignore_index = True)            
    return df

def collapse(df, fun):
    df = df.groupby(['TIME','ID']).agg(fun)
    return df

#Complex collapse 
def collapse(df, fun, w):
    """Aggregation function.

    Parameters
    ----------
    df: DataFrame
        the dataframe to aggregate
    fun:
        an aggregation function in dict format
    w:
        time window value to aggregate

    Returns
    -------
    df:
        The collapsed dataframe
    truthLabel
        The original label vector for evaluation purposes
    """
    df['TIME'] = [int(x/w) for x in df['TIME']]
    df = df.sort_values('TIME')
    truthLabel = df['LABEL']
    print("Collapse started")
    df = df.groupby(['TIME','ID']).agg(fun)
    df.columns = ['_'.join(col) for col in df.columns]
    df['distance'] = df.apply(computeDistance, axis = 1)
    df['odba'] = df.apply(computeODBA, axis = 1)
    df['speed'] = df['distance'].apply(lambda x: x/(w/1000))
    return df[cols_after], truthLabel

def resolve_time(data):
    window_len = 50
    data["TIME"] = data["TIME"].apply(lambda x: int((x - time)/(20*window_len)))
    return data

# Result in km
def computeDistance(row):

    lat1 = row['LAT_values'][0]
    lat2 = row['LAT_values'][len(row['LAT_values']) - 1]
    lon1 = row['LON_values'][0]
    lon2 = row['LON_values'][len(row['LON_values']) - 1]
#    dlon = lon2 - lon1
#    dlat = lat2 - lat1
#    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
#    c = 2 * atan2(sqrt(a), sqrt(1 - a))    
#    return R * c
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    return geopy.distance.vincenty(coords_1, coords_2).km*1000
    

#Overall dynamic body accellaration according to Wilson : https://www.3dyne.com/movebank-acceleration-viewer/documentation/calculation-of-odba/
def computeODBA(row):
    xm = np.absolute(np.asarray(row['X_AXIS_values']) - row['X_AXIS_mean'])
    ym = np.absolute(np.asarray(row['Y_AXIS_values']) - row['Y_AXIS_mean'])
    zm = np.absolute(np.asarray(row['Z_AXIS_values']) - row['Z_AXIS_mean'])
    return np.sum(xm) + np.sum(ym) + np.sum(zm)

def geod(u, v):
    coords_1 = (u[1], u[2])
    coords_2 = (v[1], v[2])
    if u[1] == 40 or v[1] == 40:
        return inf
    return vincenty(coords_1, coords_2).km*1000

def evaluation(y_test, y_pred):
    accuracy = mt.accuracy_score(y_test,y_pred)
    print("Accuracy: ", accuracy)
    f_score = mt.f1_score(y_test,y_pred, average = "weighted")
    print("F1 score: ", f_score)
    confusion_matrix = mt.confusion_matrix(y_test,y_pred)
    print("Confusion matrix: \n", confusion_matrix)
    return accuracy, f_score, confusion_matrix

def computeRealVector(pred, truth):
    label_list = truth.tolist()    
    label_list_dim = [len(x) for x in label_list]
    if len(truth) != len(pred):
        raise LabelError("\nIncorrect number of samples:\nPredicted: " + str(len(pred))+"\nReal: " + str(len(truth)))
        exit()
    vect_pred_label = [] 
    for i in range(0, len(label_list_dim)):
        vect_pred_label.append([int(pred[i])]*int(label_list_dim[i]))
    vect_pred_label = [item for sublist in vect_pred_label for item in sublist]
    return vect_pred_label

def addToResult(window, accuracy, f_score, confusion_matrix):
    global resultDict
    resultDict[str(window) +'_ACC'] = accuracy
    resultDict[str(window) +'_F1'] = f_score
    resultDict[str(window) +'_CONF'] = confusion_matrix
        
def addFinalDictToResult():
    global resultDict
    global resultDataframe
#    print(resultDict)
    resultDict = pd.Series(resultDict).to_frame().T
#    print(resultDict)
    if resultDataframe.empty:
        resultDataframe = resultDict
    else:
        resultDataframe = resultDataframe.append(resultDict, ignore_index = True)
    resultDict = {}

def mapTo(toMap, mapping):
    mapped_list = [mapping[x] for x in toMap]
    return mapTo

#%% Aggregation function
    
majority = lambda x:x.value_counts().index[0]
majority.__name__ = "majority"
majority2 = lambda x:list(x.value_counts().index)
majority2.__name__ = "majority2"
majority3 = lambda x:list(x.value_counts().values)
majority3.__name__ = "majority3"
majority4 = lambda x:sum(x.value_counts().values)
majority4.__name__ = "majority4"
values = lambda x: tuple(x)
values.__name__ = "values"
aggregateFunctions = {'COLLAR': ['mean'],
                      'ID': ['mean'],
#                          'G_LABEL': [majority, majority2],
                      'LABEL': [majority, values],
                      'LAT': ['mean', 'std', values],
                      'LON': ['mean', 'std', values],
                      'X_AXIS': ['std','max','min', 'sum', 'mean', values],
                      'Y_AXIS': ['std','max','min', 'sum', 'mean', values],
                      'Z_AXIS': ['std','max','min', 'sum', 'mean', values],
                      'TIME': ['mean']}


cols_after = ['LABEL_majority', 'LABEL_values', 'LAT_std', 'LON_std',
           'X_AXIS_std', 'X_AXIS_max', 'X_AXIS_min', 'X_AXIS_sum', 
           'X_AXIS_mean', 'Y_AXIS_std', 'Y_AXIS_max', 'Y_AXIS_min', 
           'Y_AXIS_sum', 'Y_AXIS_mean','Z_AXIS_std', 'Z_AXIS_max',
           'Z_AXIS_min', 'Z_AXIS_sum', 'Z_AXIS_mean', 'distance', 
           'odba', 'speed']

#%% Main

individual_files = glob.glob('../data/individual/*/*L.csv')

trainA = readDataset(individual_files)
trainA = trainA.drop('LABEL_F', axis =1)
trainA = trainA[trainA.LABEL_O != -1]

#%%

train = trainA.copy()
label = train.pop('LABEL_O')

#%% Majority Baseline
allMajority = [label.value_counts().index[0]]*len(label)
print("Baseline, majority classifier\n")
accuracy, f_score, confusion_matrix = evaluation(label, allMajority)

#%% Time-window
accuracy = []
f_score = []
#
windows = [180000, 120000, 100000, 80000, 60000, 50000, 30000, 20000, 10000, 5000, 1000]

for f in range(0,1):  
    trainC = trainA.copy()
    mapping = dict(enumerate(np.unique(trainC.LABEL_O)))
    print("Remapping label for XGB:")
    mapping = {v: k for k, v in mapping.items()}
    print(mapping)
    trainC.LABEL_O = trainC.LABEL_O.map(mapping)
    label = trainC.pop('LABEL_O')
    trainW, testW, train_label, test_label = ms.train_test_split(trainC, label, random_state=randint(0, f*42), test_size = 0.30)
    trainW['LABEL'] = train_label
    testW['LABEL'] = test_label
    for window in windows:
        train = trainW.copy()
        test = testW.copy()
        train, train_truth_label = collapse(train, aggregateFunctions, window)
        test, test_truth_label = collapse(test, aggregateFunctions, window)
        
        train_label = train.pop('LABEL_majority')
        test_label = test.pop('LABEL_majority')
        
        train_label_values = train.pop('LABEL_values')
        test_label_values = test.pop('LABEL_values')
        
        print('Training xgb number ' + str(f) + ' for window ' + str(window))             
        # Training model            
        dtrain = xgb.DMatrix(train, label=train_label)
        dtest = xgb.DMatrix(test, label=test_label)
        evallist  = [(dtest,'eval'), (dtrain,'train')]
        param = {}
        param['objective'] = 'multi:softmax'
        param['eta'] = 0.01
        param['max_depth'] = 100
        param['silent'] = 1
        param['nthread'] = 4
        param['num_class'] = len(np.unique(test_label)) + 1
        num_round = 100
        model = xgb.train( param, dtrain, num_round, evallist )
        y_pred = model.predict(dtest)
        accuracy, f_score, confusion_matrix = evaluation(computeRealVector(y_pred, test_label_values), test_truth_label)
        addToResult(window, accuracy, f_score, confusion_matrix)
        
    addFinalDictToResult()

resultDataframe.to_csv('individual_result_'+ str(current_milli_time())+'.csv', index = False)

#                
#        
#        
#
#
#
#
#
#
#
