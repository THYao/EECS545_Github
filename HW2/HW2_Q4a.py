# -*- coding: utf-8 -*-
"""
Created on Sat Feb 06 14:09:02 2016

@author: Shaking
"""

import numpy as np
train=np.genfromtxt("spambase.train",delimiter=",")
test=np.genfromtxt("spambase.test",delimiter=",")
print np.shape(train),np.shape(test)
train_x=train[:,0:57]
train_y=train[:,57]
test_x=test[:,0:57]
test_y=test[:,57]
whole_data=np.vstack((train_x,test_x))
print np.shape(whole_data)
median=np.median(whole_data,axis=0)
print median
trainX_proed=(train_x>=median)
testX_proed=(test_x>=median)
print np.shape(trainX_proed)
print np.shape(testX_proed)
print np.shape(train_y)
print np.shape(test_y)
train_proed=np.hstack((trainX_proed,train_y.reshape(np.shape(train_y)[0],1)))
test_proed=np.hstack((testX_proed,test_y.reshape(np.shape(test_y)[0],1)))
print np.shape(train_proed)
print np.shape(test_proed)
train_spam=train_proed[train_proed[:,57]==1]
train_ham=train_proed[train_proed[:,57]==0]
test_spam=test_proed[test_proed[:,57]==1]
test_ham=test_proed[test_proed[:,57]==0]
print np.shape(train_spam), np.shape(train_ham), np.shape(test_spam), np.shape(test_ham)
pi_spam=(np.shape(train_spam)[0]+1.0)/(np.shape(train_proed)[0]+2.0)
print pi_spam

train_spam_nonzero=[]
for i in range(len(train_spam[0,:])-1):
    nonzero=np.count_nonzero(train_spam[:,i])
    train_spam_nonzero.append(nonzero)
print train_spam_nonzero
theta_spam=(np.array(train_spam_nonzero)+1.0)/(len(train_spam)+2.0)
print theta_spam

train_ham_nonzero=[]
for i in range(len(train_ham[0,:])-1):
    nonzero=np.count_nonzero(train_ham[:,i])
    train_ham_nonzero.append(nonzero)
print train_ham_nonzero
theta_ham=(np.array(train_ham_nonzero)+1.0)/(len(train_ham)+2.0)
print theta_ham

predict_y=[]
for i in range(len(testX_proed)):
    datapoint=testX_proed[i,:]
    prob_spam=1.0
    prob_ham=1.0
    for j in range(len(datapoint)):
        if datapoint[j]==1:
            prob_spam=prob_spam*theta_spam[j]
            prob_ham=prob_spam*theta_ham[j]
        else:
            prob_spam=prob_spam*(1-theta_spam[j])
            prob_ham=prob_ham*(1-theta_ham[j])
    prob_spam=prob_spam*pi_spam
    prob_ham=prob_ham*(1-pi_spam)
    if prob_spam>prob_ham:
        predict_y.append(1)
    else:
        predict_y.append(0)
print np.shape(np.array(predict_y))

error_count=0
error_check_count=0
for i in range(len(predict_y)):
    if predict_y[i] != test_y[i]:
        error_count+=1
    if test_y[i]==1:
        error_check_count+=1

print error_count
print float(error_count)/2601

print error_check_count
print float(error_check_count)/2601