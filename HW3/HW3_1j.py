# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:12:13 2016

@author: Shaking
"""

import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt

X_train = np.genfromtxt("digits_training_data.csv",delimiter=",")
Y_train=np.genfromtxt("digits_training_labels.csv",delimiter=",")
Y_train=Y_train.reshape(np.shape(Y_train)[0],1)
trSampNum=np.shape(Y_train)[0]
featureNum=np.shape(X_train)[1]
X_test =np.genfromtxt("digits_test_data.csv",delimiter=",")
Y_test=np.genfromtxt("digits_test_labels.csv",delimiter=",")
Y_test=Y_test.reshape(np.shape(Y_test)[0],1)
tstSampNum=np.shape(Y_test)[0]
print trSampNum, featureNum
print np.shape(X_train), np.shape(Y_train)
print np.shape(X_test), np.shape(Y_test)

trainData=np.hstack((X_train,Y_train))
testData=np.hstack((X_test,Y_test))
print np.shape(trainData), np.shape(testData)

train9=trainData[trainData[:,featureNum]==9]
train4=trainData[trainData[:,featureNum]==4]
pi9=len(train9)
pi4=len(train4)
print pi9, pi4
train9_mean=np.mean(train9,axis=0)
train4_mean=np.mean(train4,axis=0)
train9_mean=train9_mean.reshape((np.shape(train9_mean)[0],1))
train4_mean=train4_mean.reshape((np.shape(train4_mean)[0],1))
print np.shape(train9_mean),np.shape(train4_mean)
train9_XMean=train9_mean[0:featureNum,:]
train4_XMean=train4_mean[0:featureNum,:]
print np.shape(train9_XMean), np.shape(train4_XMean)
CovMatr=np.cov(trainData[:,0:featureNum].T)
InvCov=np.linalg.pinv(CovMatr)
print np.shape(CovMatr)

y_predTr=[]
for i in range(trSampNum):
    value9=np.dot(np.dot(train9_XMean.T,InvCov),X_train[i,:])-0.5*np.dot(np.dot(train9_XMean.T,InvCov),train9_XMean)+np.log(pi9)
    value4=np.dot(np.dot(train4_XMean.T,InvCov),X_train[i,:])-0.5*np.dot(np.dot(train4_XMean.T,InvCov),train4_XMean)+np.log(pi4)
    if value9>=value4:
        y_predTr.append(9)
    else:
        y_predTr.append(4)

y_predTst=[]
for i in range(tstSampNum):
    value9=np.dot(np.dot(train9_XMean.T,InvCov),X_test[i,:])-0.5*np.dot(np.dot(train9_XMean.T,InvCov),train9_XMean)+np.log(pi9)
    value4=np.dot(np.dot(train4_XMean.T,InvCov),X_test[i,:])-0.5*np.dot(np.dot(train4_XMean.T,InvCov),train4_XMean)+np.log(pi4)
    if value9>=value4:
        y_predTst.append(9)
    else:
        y_predTst.append(4)

def get_accu(pred,target):
    error_lst=[]
    count=0
    for i in range(len(pred)):
        if pred[i]==target[i]:
            count+=1
        else:
            error_lst.append(i)
    return (float(count)/len(pred),error_lst)

Y_trainErrLst=get_accu(y_predTr,Y_train)[1]
Y_testErrLst=get_accu(y_predTst,Y_test)[1]
print get_accu(y_predTr,Y_train)
print get_accu(y_predTst,Y_test)

for i in Y_trainErrLst:
    plt.imshow(X_train[i].reshape((26,26)), interpolation="nearest", cmap=cm.Greys_r)
    plt.show()

for i in Y_testErrLst:
    plt.imshow(X_test[i].reshape((26,26)), interpolation="nearest", cmap=cm.Greys_r)
    plt.show()