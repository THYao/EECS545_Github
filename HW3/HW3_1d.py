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

def Gradient(Weight, Bias, Cost, feature, label):
    sumE=np.zeros(np.shape(Weight))
    for i in range(len(label)):
        feature_T=feature[i].reshape((np.shape(feature[i])[0],1))
        sumE+=(label[i]*Indicator(Weight, Bias, feature[i],label[i])*feature_T)
    Wgrad=Weight-Cost*sumE
    sumB=0
    for i in range(len(label)):
        sumB+=(label[i]*Indicator(Weight, Bias, feature[i],label[i]))
    Bgrad=-Cost*sumB
    return (Wgrad,Bgrad)

def update(Weight, Bias, Iter_i, eta, gradient):
    alpha_i=float(eta)/(1+Iter_i*eta)
    Weight=Weight-alpha_i*gradient[0]
    Bias=Bias-alpha_i*gradient[1]
    return (Weight, Bias)
ite_lst=[]
Trainaccu_lstBGD=[]
for i in range(1,100):
    gradient=Gradient(weight, bias, 3, train_feature, train_label)
    weight=update(weight, bias, i, 0.001, gradient)[0]
    bias=update(weight, bias, i, 0.001, gradient)[1]
    Trainaccu=get_Trainaccu(weight, bias, train_feature, train_label)
    ite_lst.append(i)
    Trainaccu_lstBGD.append(Trainaccu)
print weight, np.shape(weight)
print bias, np.shape(bias)
print get_Trainaccu(weight,bias,test_feature, test_label)

plt.plot(ite_lst,Trainaccu_lstBGD,label="Batch Gradient Descent")
plt.xlabel("iteration")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
