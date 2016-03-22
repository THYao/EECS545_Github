# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 00:47:02 2016

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

train_feature=scipy.sparse.csr_matrix(trainFeature).toarray()
test_feature=scipy.sparse.csr_matrix(testFeature).toarray()
print np.shape(train_feature)
print np.shape(test_feature)

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
print np.shape(train_y)
print np.shape(train_feature)

train_data=np.hstack((train_feature,train_y))
print np.shape(train_data)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
y_pred = mnb.fit(train_feature, train_y).predict(test_feature)
print y_pred, len(y_pred)

g=open('HW2_Q4b.csv','w')
csv_w=csv.writer(g)
csv_w.writerow(y_pred)
g.close()

