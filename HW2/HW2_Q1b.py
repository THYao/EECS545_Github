# -*- coding: utf-8 -*-
"""
Created on Sat Feb 06 23:22:52 2016

@author: Shaking
"""
from matplotlib import pyplot as plt
import numpy as np
def gaussian_kernel(x_i, query_i, decaySpeed):
    diff = x_i - query_i
    distance=np.dot(np.transpose(diff),diff)
    return np.exp(-distance/(2*np.square(decaySpeed)))

def weight_matrix(xMatrix, query_i, decaySpeed):
    weight=np.eye(np.shape(xMatrix)[0])
    for i in range(np.shape(xMatrix)[0]):
        weight[i,i]=gaussian_kernel(xMatrix[i,:],query_i,decaySpeed)
    return weight
'''
data_x=np.array([[1.1,2.2,3.2,4.2,5.2],[2.3,3.3,4.3,5.3,6.3],[3.4,4.4,5.4,6.4,7.4]])
data_y=np.array([10,23,53])
print np.shape(data_x)
query=np.array([[1.1,4.1,6.1,7.1,3.1],[2.2,5.1,7.1,2.2,6.1]])
for i in range(len(query)):
    R_w=weight_matrix(data_x,query[i],1)
    weighted_x=np.dot(np.dot(np.transpose(data_x),R_w),data_x)
    coeff=np.dot(np.dot(np.dot(np.linalg.inv(weighted_x),np.transpose(data_x)),R_w),data_y)
    print coeff
'''
train=np.genfromtxt('train_graphs_f16_autopilot_cruise.csv',delimiter=',')
test=np.genfromtxt('test_locreg_f16_autopilot_cruise.csv',delimiter=',')
train=np.delete(train,0,0)
test=np.delete(test,0,0)
train_y=train[:,7]
train_x=train[:,1:7]
one_train=np.ones(np.shape(train_x[:,0]))
#train_x=np.hstack((one_train.reshape(np.shape(train_x)[0],1),train_x))
train_x_mean=np.mean(train_x,axis=0)
train_x_variance=np.var(train_x,axis=0)
print train_x_mean, train_x_variance
train_x_normed=(train_x-train_x_mean)/np.sqrt(train_x_variance)
test_y=test[:,7]
test_x=test[:,1:7]
one_test=np.ones(np.shape(test_x[:,0]))
#test_x=np.hstack((one_test.reshape(np.shape(test_x)[0],1),test_x))
test_x_mean=np.mean(test_x,axis=0)
test_x_variance=np.var(test_x,axis=0)
print test_x_mean, test_x_variance
test_x_normed=(test_x-test_x_mean)/np.sqrt(test_x_variance)
tau=np.logspace(-2, 1, num=10, base=2)
print tau
RMSE_lst=[]

for i in range(len(tau)):
    RMSE=0
    for j in range(len(test_x_normed)):
        R_w_j=weight_matrix(train_x_normed,test_x_normed[j],tau[i])
        weighted_x=np.dot(np.dot(np.transpose(train_x_normed),R_w_j),train_x_normed)
        coeff_j=np.dot(np.dot(np.dot(np.linalg.inv(weighted_x),np.transpose(train_x_normed)),R_w_j),train_y)
        predict_y_j=np.dot(np.transpose(coeff_j),test_x_normed[j])
        RMSE+=np.sqrt((2*((predict_y_j-test_y[j])**2)/100))
    RMSE_lst.append(RMSE)

print RMSE_lst
plt.plot(tau,RMSE_lst,'-r',label="RMSE")
plt.xlabel(r'$\tau$')
plt.ylabel("RMSE")
plt.legend()
plt.show()
