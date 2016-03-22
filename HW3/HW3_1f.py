# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 02:25:45 2016

@author: Shaking
"""

import numpy as np
from matplotlib import pyplot as plt
train_feature=np.genfromtxt("digits_training_data.csv",delimiter=",")
train_label=np.genfromtxt("digits_training_labels.csv",delimiter=",")
train_label=train_label.reshape((np.shape(train_label)[0],1))
for i in range(len(train_label)):
    if train_label[i]==9:
        train_label[i]=1
    else:
        train_label[i]=-1
train_data=np.hstack((train_feature, train_label))
test_feature=np.genfromtxt("digits_test_data.csv",delimiter=",")
test_label=np.genfromtxt("digits_test_labels.csv",delimiter=",")
test_label=test_label.reshape((np.shape(test_label)[0],1))
for i in range(len(test_label)):
    if test_label[i]==9:
        test_label[i]=1
    else:
        test_label[i]=-1
test_data=np.hstack((test_feature, test_label))

weight=np.zeros((np.shape(train_feature)[1],1))
bias=0

def get_Trainaccu(Weight, b, feature, label):
    predicted=np.dot(feature, Weight)
    b_ones=b*np.ones(np.shape(predicted))
    predicted+=b_ones
    count=0
    for i in range(len(predicted)):
        if predicted[i]>=0 and label[i]==1:
            count+=1
        if predicted[i]<0 and label[i]==-1:
            count+=1
    return float(count)/len(predicted)

print get_Trainaccu(weight, bias, train_feature, train_label)

def Indicator(Weight, Bias, feature_i, label_i):
    if label_i*(np.dot(Weight.T,feature_i)+Bias)<1:
        return 1
    else:
        return 0

print Indicator(weight, bias, train_feature[0,:],train_label[0])

def Gradient_i(Weight, Bias, Cost, feature_i, label_i, N):
    feature_T=feature_i.reshape((np.shape(feature_i)[0],1))
    I_i=Indicator(Weight, Bias, feature_i, label_i)
    Wgrad=Weight/float(N)-Cost*label_i*I_i*feature_T
    Bgrad=-Cost*label_i*I_i
    return (Wgrad,Bgrad)

def update(Weight, Bias, Iter_i, eta, gradient):
    alpha_i=float(eta)/(1+Iter_i*eta)
    Weight=Weight-alpha_i*gradient[0]
    Bias=Bias-alpha_i*gradient[1]
    return (Weight, Bias)
ite_lst=[]
Trainaccu_lstSGD=[]
for i in range(1,100):
    sample_Num=len(train_label)
    for j in np.random.choice(sample_Num,sample_Num,replace=False):
        gradient=Gradient_i(weight, bias, 3, train_feature[j], train_label[j],sample_Num)
        weight=update(weight, bias, i, 0.001, gradient)[0]
        bias=update(weight, bias, i, 0.001, gradient)[1]
    Trainaccu=get_Trainaccu(weight, bias, train_feature, train_label)
    ite_lst.append(i)
    Trainaccu_lstSGD.append(Trainaccu)
    
print ite_lst
print Trainaccu_lstSGD
plt.plot(ite_lst,Trainaccu_lstSGD,label="Stochastic Gradient Descent")
plt.plot(ite_lst,Trainaccu_lstBGD,label="Batch Gradient Descent")
plt.xlabel("iteration")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
