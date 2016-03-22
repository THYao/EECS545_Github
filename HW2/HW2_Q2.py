# -*- coding: utf-8 -*-
"""
Created on Fri Feb 05 16:34:57 2016

@author: Shaking
"""
import csv
import numpy as np
train=np.genfromtxt("steel_composition_train.csv",delimiter=",")
train=np.delete(train,0,1)
train=np.delete(train,0,0)
test_csv=np.genfromtxt("steel_composition_test.csv",delimiter=",")
test_csv=np.delete(test_csv,0,0)
test_feature_noOne=test_csv[:,1:9]
#feature and y extracted
feature_noOne=train[:,0:8]
train_y=train[:,8]

def AddColumn(Matrix1,Matrix2):
    dimM1=np.shape(Matrix1)
    dimM2=np.shape(Matrix2)
    if len(dimM1)==1:
        dimM1+=(1,)
    if len(dimM2)==1:
        dimM2+=(1,)
    BigMatrix_flat=np.append(np.transpose(Matrix1),np.transpose(Matrix2))
    BigMatrix=BigMatrix_flat.reshape(dimM2[1]+dimM1[1],dimM1[0])
    BigMatrix=np.transpose(BigMatrix)
    return BigMatrix

RMSE_train_1a=[]
RMSE_test_1a=[]
M_lst=range(1,5)
for i in M_lst:
    one=np.ones(np.shape(train[:,0]))
    test_one=np.ones(np.shape(test_csv[:,0]))
    if i ==1:
        Phi_x=AddColumn(one,feature_noOne)
        test_Phi=AddColumn(test_one,test_feature_noOne)
    else:
        Phi_x=AddColumn(Phi_x, np.power(feature_noOne,i))
        test_Phi=AddColumn(test_Phi, np.power(test_feature_noOne,i))
    Phi_dagger=np.dot(np.linalg.inv(np.dot(np.transpose(Phi_x),Phi_x)),np.transpose(Phi_x))
    print np.shape(Phi_dagger)    
    w_1=np.dot(Phi_dagger,train_y)
    print np.shape(w_1), w_1

predict_y=np.dot(test_Phi,w_1)
print predict_y
f=open('HW2_Q2.csv','w')
csv_w=csv.writer(f)
csv_w.writerow(predict_y)
f.close()