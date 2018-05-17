import re

import sklearn.metrics as mt
import collections


#%% Functions

def evaluation(y_test, y_pred):
    accuracy = mt.accuracy_score(y_test,y_pred)
    print("Accuracy: ", accuracy)
    f_score = mt.f1_score(y_test,y_pred, average = "weighted")
    print("F1 score: ", f_score)
    confusion_matrix = mt.confusion_matrix(y_test,y_pred)
    print("Confusion matrix: \n", confusion_matrix)
    return accuracy, f_score, confusion_matrix

def parseOutput(file):
    lines = open(file).readlines()
    regex = re.compile(r'^[0-9]+ [0-9]+ [0-9]+$')
    results = list(filter(regex.search, lines))

    y_true = []
    y_pred = []

    for r in results:
        splits = r.split(" ")
        y_true.append(splits[1])
        y_pred.append(splits[2])
        
    y_pred = [x.replace('\n', '') for x in y_pred]

    return y_true, y_pred


true, pred =parseOutput('output.txt')

evaluation(true,pred)
print(collections.Counter(true))
print(collections.Counter(pred))