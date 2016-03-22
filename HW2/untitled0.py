# -*- coding: utf-8 -*-
"""
Created on Sat Feb 06 14:09:02 2016

@author: Shaking
"""
import numpy as np
import pickle
import scipy
import csv

with open('trainFeatures.pkl', 'rb') as f:
    trainFeature = pickle.load(f)
with open('testFeatures.pkl', 'rb') as g:
    testFeature=pickle.load(g)

train_x=scipy.sparse.csr_matrix(trainFeature).toarray()
test_x=scipy.sparse.csr_matrix(testFeature).toarray()
print np.shape(train_x)
print np.shape(test_x)

f=open("spam_filter_train.txt",'r')
train_y=[]
for line in f:
    if line.split()[0]=="spam":
        train_y.append(1)
    else:
        train_y.append(0)
f.close()
print len(train_y)
train_y=np.array(train_y).reshape((len(train_y),1))
train_data=np.hstack((train_x,train_y))


train_spam=train_data[train_data[:,47922]==1]
train_ham=train_data[train_data[:,47922]==0]
print np.shape(train_spam), np.shape(train_ham)
pi_spam=(np.shape(train_spam)[0]+1.0)/(np.shape(train_data)[0]+2.0)
print pi_spam

train_spam_nonzero=[]
for i in range(len(train_spam[0,:])-1):
    nonzero=np.count_nonzero(train_spam[:,i])
    train_spam_nonzero.append(nonzero)
#print train_spam_nonzero
theta_spam=(np.array(train_spam_nonzero)+1.0)/(len(train_spam)+2.0)
#print theta_spam

train_ham_nonzero=[]
for i in range(len(train_ham[0,:])-1):
    nonzero=np.count_nonzero(train_ham[:,i])
    train_ham_nonzero.append(nonzero)
#print train_ham_nonzero
theta_ham=(np.array(train_ham_nonzero)+1.0)/(len(train_ham)+2.0)
#print theta_ham

predict_y=[]
for i in range(len(test_x)):
    datapoint=test_x[i,:]
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

g=open('HW2_Q4b_try.csv','w')
csv_w=csv.writer(g)
csv_w.writerow(predict_y)
g.close()