# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:56:47 2016

@author: Shaking
"""
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.cm as cm
from matplotlib import pyplot as plt

# Read in csv file and conver to a numpy array 
data = np.genfromtxt('./digits_training_data.csv', delimiter=',')

# plot a random training image (row)
k = int(np.random.random()*data.shape[0])
plt.imshow(data[k].reshape((26,26)), interpolation="nearest", cmap=cm.Greys_r)


X_train = pd.read_csv("digits_training_data.csv")
Y_train=pd.read_csv("digits_training_labels.csv")
X_test = pd.read_csv("digits_test_data.csv")
Y_test=pd.read_csv("digits_test_labels.csv")

clfTst_svm = svm.SVC(kernel="rbf", C=100, gamma=1e-7)
clfTst_svm.fit(X_train, Y_train)
y_predTst_svm = clfTst_svm.predict(X_test)
acc_svm = accuracy_score(Y_test, y_predTst_svm)
print "SVM accuracy: ",acc_svm

clfTr_svm = svm.SVC(kernel="rbf", C=100, gamma=1e-7)
clfTr_svm.fit(X_train, Y_train)
y_predTr_svm = clfTst_svm.predict(X_train)
acc_svm = accuracy_score(Y_train, y_predTr_svm)
print "SVM accuracy: ",acc_svm

test=np.array(Y_test.values)
test_lst=[]
for i in range(len(test)):
    test_lst.append(test[i][0])
print test_lst

count=0
error_lst=[]
for i in range(len(test_lst)):
    if test_lst[i]==y_predTst_svm[i]:
        count+=1
    else:
        error_lst.append(i)
print float(count)/len(test_lst)
print error_lst

drawX_test = np.genfromtxt('./digits_test_data.csv', delimiter=',')
for i in error_lst:
    plt.imshow(drawX_test[i].reshape((26,26)), interpolation="nearest", cmap=cm.Greys_r)
    plt.show()