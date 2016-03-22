# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 22:42:50 2016

@author: Shaking
"""

import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
trainLabels = np.loadtxt('trainingLabels.gz', dtype=np.uint8, delimiter=',')
trainData = np.loadtxt('trainingData.gz', dtype=np.uint8, delimiter=',')
testData = np.loadtxt('testData.gz',dtype=np.uint8,delimiter=',')
clf_rf = RandomForestClassifier()
clf_rf.fit(trainData, trainLabels)
y_pred_rf = clf_rf.predict(testData)
np.savetxt('pred.csv',y_pred_rf,delimiter=',')