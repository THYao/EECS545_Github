# -*- coding: utf-8 -*-
"""
Created on Fri Feb 05 16:34:57 2016

@author: Shaking
"""
from matplotlib import pyplot as plt
import numpy as np
train=np.genfromtxt("train_graphs_f16_autopilot_cruise.csv",delimiter=",")
train=np.delete(train,0,1)
train=np.delete(train,0,0)
test_csv=np.genfromtxt("test_graphs_f16_autopilot_cruise.csv",delimiter=",")
test_csv=np.delete(test_csv,0,0)
test_feature_noOne=test_csv[:,1:7]
test_y=test_csv[:,7]
#feature and y extracted
feature_noOne=train[:,0:6]
train_y=train[:,6]

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
M_lst=range(1,7)
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
    Error_train=0.5*np.dot(np.transpose(np.dot(Phi_x,w_1)),np.dot(Phi_x,w_1))-np.dot(np.transpose(np.dot(Phi_x,w_1)),train_y)+0.5*np.dot(np.transpose(train_y),train_y)
    RMSE_train=(2*Error_train/3426)**(0.5)
    print np.shape(np.transpose(np.dot(Phi_x,w_1))),np.shape(test_y)
    Error_test=0.5*np.dot(np.transpose(np.dot(test_Phi,w_1)),np.dot(test_Phi,w_1))-np.dot(np.transpose(np.dot(test_Phi,w_1)),test_y)+0.5*np.dot(np.transpose(test_y),test_y)
    RMSE_test=(2*Error_test/np.shape(test_y)[0])**(0.5)
    RMSE_train_1a.append(RMSE_train)
    RMSE_test_1a.append(RMSE_test)
print RMSE_train_1a
print RMSE_test_1a

plt.plot(M_lst,RMSE_train_1a,'-r',label='train RMSE')
plt.plot(M_lst,RMSE_test_1a,'-b', label='test RMSE')
plt.xlabel('M')
plt.ylabel('RMSE')
plt.legend()
plt.show()

RMSE_train_1b=[]
RMSE_test_1b=[]    
for j in range(-40,21):
    Lambda=np.exp(j)
    Phi_reg_dagger=np.dot(np.linalg.inv(Lambda*np.eye(np.shape(Phi_x)[1])+np.dot(np.transpose(Phi_x),Phi_x)),np.transpose(Phi_x))
    w_reg=np.dot(Phi_reg_dagger,train_y)
    Error_train=0.5*np.dot(np.transpose(np.dot(Phi_x,w_reg)),np.dot(Phi_x,w_reg))-np.dot(np.transpose(np.dot(Phi_x,w_reg)),train_y)+0.5*np.dot(np.transpose(train_y),train_y)+0.5*Lambda*np.dot(np.transpose(w_reg),w_reg)
    RMSE_train=(2*Error_train/3426)**(0.5)
    Error_test=0.5*np.dot(np.transpose(np.dot(test_Phi,w_reg)),np.dot(test_Phi,w_reg))-np.dot(np.transpose(np.dot(test_Phi,w_reg)),test_y)+0.5*np.dot(np.transpose(test_y),test_y)+0.5*Lambda*np.dot(np.transpose(w_reg),w_reg)
    RMSE_test=(2*Error_test/np.shape(test_y)[0])**(0.5)
    RMSE_train_1b.append(RMSE_train)
    RMSE_test_1b.append(RMSE_test)
print RMSE_train_1b
print RMSE_test_1b
 
plt.plot(range(-40,21),RMSE_train_1b, '-r',label= 'train RMSE')
plt.plot(range(-40,21),RMSE_test_1b,'-b', label="tets RMSE")   
plt.xlabel(r'ln($\lambda$)')
plt.ylabel('RMSE')
plt.legend()
plt.show()