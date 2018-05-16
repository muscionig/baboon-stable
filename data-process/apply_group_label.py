# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:36:39 2018

@author: Guido Muscioni

Description: This script can be used to applying the group label to the days 3 and 4.
An observation is removed at the end for syncronization purposes.
Notice that no individual labels overlap with the group label.
"""

import pandas as pd
import numpy as np
import glob

def getVectLabel(df_label, sampling_rate):
    df_label = df_label.drop(df_label.columns[0], axis =1)
    vectorizedLabel = df_label[df_label.columns[0]]. values
    vectorizedLabel = vectorizedLabel[:len(vectorizedLabel)-1]
    newLabel = np.repeat(vectorizedLabel, sampling_rate)
    return newLabel
    

sampling_rate = 12
days = {3: '0803', 4:'0804'}
for day in days.keys():
    df_label = pd.read_csv('../data/group/group_label_'+str(day)+'.csv', header = None)
    g_label = getVectLabel(df_label, sampling_rate)
    g_label = g_label.astype(int)
    files = glob.glob('../data/group/'+days[day]+"/*.csv")
    for file in files:
        df = pd.read_csv(file)
        df['LABEL_GROUP'] = g_label
        df.to_csv(file, index = False)
        
