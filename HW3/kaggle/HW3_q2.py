# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 19:00:13 2016

@author: Shaking
"""

import numpy as np
from sklearn.svm import LinearSVC
import csv
from load_data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

y_train = np.loadtxt('trainingLabels.gz', dtype=np.uint8, delimiter=',')
X_train = np.loadtxt('trainingData.gz', dtype=np.uint8, delimiter=',')
X_test = np.loadtxt('testData.gz', dtype=np.uint8, delimiter=',')

imgHeight = np.sqrt(X_train.shape[1])
show_image(X_train[1].reshape((imgHeight, imgHeight)))

clf_svm = LinearSVC()
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)

f=open("HW3Kaggle_1.csv",'w')
writer=csv.writer(f)
writer.writerows(["id","category"])
f.close()
