# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:14:01 2018

@author: Guido Muscioni

Description: 
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sklearn.utils.class_weight as cwc
import sklearn.metrics as mt
import xgboost as xgb
import glob 
from sklearn import model_selection as ms
import ast
import collections
import time
from random import randint

#%% Globals 
nameAll = ['LOGISTIC_ACC_ALL',
 'LOGISTIC_F1_ALL',
 'LOGISTIC_CONF_ALL',
 'TREE_G_ACC_ALL',
 'TREE_G_F1_ALL',
 'TREE_G_CONF_ALL',
 'TREE_E_ACC_ALL',
 'TREE_E_F1_ALL',
 'TREE_E_CONF_ALL',
 'TREE_G_CONF_ALL',
 'RANDF_ACC_ALL',
 'RANDF_F1_ALL',
 'RANDF_CONF_ALL',
 'XGB_ACC_ALL',
 'XGB_F1_ALL',
 'XGB_CONF_ALL']

nameRaw = ['LOGISTIC_ACC_RAW',
 'LOGISTIC_F1_RAW',
 'LOGISTIC_CONF_RAW',
 'TREE_G_ACC_RAW',
 'TREE_G_F1_RAW',
 'TREE_G_CONF_RAW',
 'TREE_E_ACC_RAW',
 'TREE_E_F1_RAW',
 'TREE_E_CONF_RAW',
 'TREE_G_CONF_RAW',
 'RANDF_ACC_RAW',
 'RANDF_F1_RAW',
 'RANDF_CONF_RAW',
 'XGB_ACC_RAW',
 'XGB_F1_RAW',
 'XGB_CONF_RAW']

nameAgg = ['LOGISTIC_ACC_AGG',
 'LOGISTIC_F1_AGG',
 'LOGISTIC_CONF_AGG',
 'TREE_G_ACC_AGG',
 'TREE_G_F1_AGG',
 'TREE_G_CONF_AGG',
 'TREE_E_ACC_AGG',
 'TREE_E_F1_AGG',
 'TREE_E_CONF_AGG',
 'TREE_G_CONF_AGG',
 'RANDF_ACC_AGG',
 'RANDF_F1_AGG',
 'RANDF_CONF_AGG',
 'XGB_ACC_AGG',
 'XGB_F1_AGG',
 'XGB_CONF_AGG']

nameNet = ['LOGISTIC_ACC_NET',
 'LOGISTIC_F1_NET',
 'LOGISTIC_CONF_NET',
 'TREE_G_ACC_NET',
 'TREE_G_F1_NET',
 'TREE_G_CONF_NET',
 'TREE_E_ACC_NET',
 'TREE_E_F1_NET',
 'TREE_E_CONF_NET',
 'TREE_G_CONF_NET',
 'RANDF_ACC_NET',
 'RANDF_F1_NET',
 'RANDF_CONF_NET',
 'XGB_ACC_NET',
 'XGB_F1_NET',
 'XGB_CONF_NET']

nameMaj = ['MAJ_ACC_',
 'MAJ_F1_',
 'MAJ_CONF_']

column_name = ['WINDOW'] + nameAll + nameRaw + nameAgg + nameNet + nameMaj
global resultDataframe
resultDataframe = pd.DataFrame(columns=column_name)

global resultDict
resultDict = {}

#%% Functions

current_milli_time = lambda: int(round(time.time() * 1000))

def readDataset(fileList):
    print("Reading")
    df = pd.DataFrame()
    for file in fileList:
#        print("Reading: " + file)
        temp = pd.read_csv(file, sep = ',')
        temp = temp.fillna(method = 'ffill')
#        temp.columns = ['TIME', 'ID', 'COLLAR', 'LAT', 'LON', 'X_AXIS', 'Y_AXIS', 'Z_AXIS','G_LABEL']
        temp = temp.dropna()
        if(df.empty):
#            if 'LABEL_O' not in list(temp.columns):
#                temp['LABEL_O'] = -1
#                temp['LABEL_F'] = -1
            df = temp.copy()
        else:
#            if 'LABEL_O' not in list(temp.columns):
#                temp['LABEL_O'] = -1
#                temp['LABEL_F'] = -1
            df = df.append(temp.copy(), ignore_index = True)
    print("End Reading")      
    return df

def computeClassWeight(y_train):
    classes = np.unique(y_train) 
    cw = cwc.compute_class_weight("balanced", classes, y_train)
    cw = professions_dict = dict(zip(classes, cw))
    return cw

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

def addToResult(features, model, accuracy, f_score, confusion_matrix):
    global resultDict
    resultDict[model + '_ACC_' + features] = accuracy
    resultDict[model + '_F1_' + features] = f_score
    resultDict[model + '_CONF_' + features] = confusion_matrix
    
def addFinalDictToResult(window):
    global resultDict
    global resultDataframe
    resultDict['WINDOW'] = window
#    print(resultDict)
    resultDict = pd.Series(resultDict).to_frame().T
#    print(resultDict)
    if resultDataframe.empty:
        resultDataframe = resultDict
    else:
        resultDataframe = resultDataframe.append(resultDict, ignore_index = True)
    resultDict = {}

    

#%%    
    
windows = [180000, 120000, 100000, 80000, 60000, 50000, 30000, 20000, 10000, 5000, 1000]
for window in windows:
    fileList = glob.glob('../data/network/complete/'+ str(window)+ '/*Network*C.csv')
    
    #%%
    
    data = readDataset(fileList)
    data = data.drop(['LAT_mean','LON_mean', 'ID', 'TIME', 'NEIGHBORS'], axis =1)
    data = data.set_index('indexS')
    dataSaved = data.copy()
    label = data.pop('G_LABEL_majority')
    
    train, test, train_label, test_label = ms.train_test_split(data, label, random_state=42, test_size = 0.30)  
    train.is_copy = False
    test.is_copy = False
    train['LABEL'] = train_label
    test['LABEL'] = test_label
    
    trainSaved = train.copy()
    testSaved = test.copy()
    
    real_label = test.pop('G_LABEL_values').apply(lambda x: list(ast.literal_eval(x)))
    real_label_dimension = [len(x) for x in real_label]
    real_label = [item for sublist in real_label for item in sublist]
    
    allMajority = [train.LABEL.value_counts().index[0]]*len(real_label)
    print("Baseline, majority classifier\n")
    accuracy, f_score, confusion_matrix = evaluation(real_label, allMajority)
    features = ''
    model = 'MAJ'
    addToResult(features, model, accuracy, f_score, confusion_matrix)
    
    #%%
    
    print("All features for window: " + str(window))
    features = 'ALL'
    
    #%%
    
    print("\nLogistic using the entire dataset on a 30% holdout: \n")
    #--------------- Recover dataset -------------------
    train = trainSaved.copy()
    train_label = train.pop('LABEL')
    test = testSaved.copy()
    test_label = test.pop('LABEL')
    
    train.pop('G_LABEL_values')
    
    real_label = test.pop('G_LABEL_values').apply(lambda x: list(ast.literal_eval(x)))
    real_label_dimension = [len(x) for x in real_label]
    real_label = [item for sublist in real_label for item in sublist]
    
    #---------------------------------------------------
    cw = computeClassWeight(train_label)
    log = LogisticRegression( penalty = 'l1', C = 0.1)
    log.fit(train, train_label)
    y_pred = log.predict(test)
    
    newLabelV = []
    for i in range(0, len(real_label_dimension)):
        newLabelV.append([int(y_pred[i])]*int(real_label_dimension[i]))
    new_pred = [item for sublist in newLabelV for item in sublist]
    
    accuracy, f_score, confusion_matrix = evaluation(real_label, new_pred)
    model = 'LOGISTIC'
    addToResult(features, model, accuracy, f_score, confusion_matrix)
    
    #%%
    
    print("\nDecision Tree (gini) the entire dataset on a 30% holdout: \n")
    #--------------- Recover dataset -------------------
    train = trainSaved.copy()
    train_label = train.pop('LABEL')
    test = testSaved.copy()
    test_label = test.pop('LABEL')
    
    train.pop('G_LABEL_values')
    
    real_label = test.pop('G_LABEL_values').apply(lambda x: list(ast.literal_eval(x)))
    real_label_dimension = [len(x) for x in real_label]
    real_label = [item for sublist in real_label for item in sublist]
    #---------------------------------------------------
    tree = DecisionTreeClassifier(criterion = "gini")
    tree.fit(train, train_label)
    y_pred = tree.predict(test)
    
    newLabelV = []
    for i in range(0, len(real_label_dimension)):
        newLabelV.append([int(y_pred[i])]*int(real_label_dimension[i]))
    new_pred = [item for sublist in newLabelV for item in sublist]
    
    accuracy, f_score, confusion_matrix = evaluation(real_label, new_pred)
    model = 'TREE_G'
    addToResult(features, model, accuracy, f_score, confusion_matrix)
    
    #%%
    
    print("\n eXtreme Gradient Boosting on a 25% bootstrapping: \n")
    
    train = trainSaved.copy().append(testSaved.copy())
    mapping = dict(enumerate(np.unique(train.LABEL)))
    print("Remapping label for XGB:")
    mapping = {v: k for k, v in mapping.items()}
    print(mapping)
    train.LABEL = train.LABEL.map(mapping)
    test = train.pop('LABEL')
    accuracyL = []
    f1L = []
    f1_max = 0
    for i in range(10):
        random = i*42
        print("XGBoost iteration number: ", i+1)
        X_train, X_test, y_train, y_test  = ms.train_test_split(train, test, random_state=randint(0, i*42), test_size = 0.25)
        cw = computeClassWeight(y_train)
        
        train_label_values = X_train.pop('G_LABEL_values')
        test_label_values = X_test.pop('G_LABEL_values').apply(lambda x: list(ast.literal_eval(x)))
        
        #train
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        evallist  = [(dtest,'eval'), (dtrain,'train')]
        param = {}
    #    param['class_weight'] = cw
        param['objective'] = 'multi:softmax'
        param['eta'] = 0.1
        param['max_depth'] = 100
        param['silent'] = 1
        param['nthread'] = 4
        param['num_class'] = len(np.unique(test))+1
        param['eval_metric'] = 'merror'
        num_round = 20
        model = xgb.train( param, dtrain, num_round, evallist, verbose_eval =False)
        y_pred = model.predict(dtest)
        
        #create the correct vector for evaluation
        test_truth_label = [item for sublist in test_label_values for item in sublist]
        for k in np.unique(test_truth_label):
            if k not in list(mapping.keys()):
                if k == -1.0:
                    mapping[k] = 2.0
                else:
                    m = max(mapping, key=mapping.get)
                    mapping[k] = mapping[m]+1
        test_truth_label = [mapping[x] if x in list(mapping.keys()) else x for x in test_truth_label]
        test_truth_label = [2.0 if x == -1.0 else x for x in test_truth_label]
        
        #evaluation
        accuracy, f_score, confusion_matrix = evaluation(computeRealVector(y_pred, test_label_values), test_truth_label)
        accuracyL.append(accuracy)
        f1L.append(f_score)
        
        #save best confusion matrix
        if f_score > f1_max :
            confusion_max = confusion_matrix
            f1_max = f_score
        
    print("Average accuracy over 10 iterations: ", np.mean(accuracyL), np.std(accuracyL))
    print("Average f1 score over 10 iterations: ", np.mean(f1L), np.std(f1L))
    print("Max accuracy over 10 iterations: ", np.max(accuracyL))
    print("Max f1 score over 10 iterations: ", np.max(f1L))
    model = 'XGB'
    addToResult(features, model, np.max(accuracyL), np.max(f1L), confusion_max)
    
    toSee = addFinalDictToResult(window)

resultDataframe.to_csv('complex_individual_result_'+ str(current_milli_time())+'.csv', index = False)




