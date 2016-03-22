# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 05:56:10 2016

@author: Shaking
"""
import numpy as np
trD_raw=np.genfromtxt("steel_composition_train.csv",delimiter=",")
trD_raw=trD_raw[1:,1:]
featureNum=np.shape(trD_raw)[1]-1
sampleNum=np.shape(trD_raw)[0]
print featureNum
print sampleNum
trD_mean=np.mean(trD_raw,axis=0)
trD_var=np.var(trD_raw,axis=0)
trD=(trD_raw-trD_mean)/np.sqrt(trD_var)
trF=trD[:,0:featureNum]
trL=trD[:,featureNum]
trL_nonNorm=trD_raw[:,featureNum]

def poly_kernelFun(Samp1, Samp2, power):
    value=(np.dot(Samp1,Samp2)+1)**power
    return value

def Gauker(Samp1, Samp2):
    dist=np.dot(Samp1-Samp2,Samp1-Samp2)
    value=np.exp(-dist/2)
    return value

def Poly_Gram_Mtr(Feature1, Feature2, power):
    SamNum=len(Feature1)
    GramMat=np.zeros((SamNum,SamNum))
    for i in range(SamNum):
        for j in range(SamNum):
            GramMat[i,j]=poly_kernelFun(Feature1[i,:],Feature2[j,:],power)
    return GramMat

def Gau_Gram_Mtr(Feature1, Feature2):
    SamNum=len(Feature1)
    GramMat=np.zeros((SamNum,SamNum))
    for i in range(SamNum):
        for j in range(SamNum):
            GramMat[i,j]=Gauker(Feature1[i,:],Feature2[j,:])
    return GramMat

def Gram_Vec(Feature, queryPt, kerPow):
    SampNum=len(Feature)
    GramVec=np.zeros((SampNum,1))
    if kerPow==0:
        for i in range(SampNum):
            GramVec[i,0]=Gauker(Feature[i],queryPt)
    else:
        for i in range(SampNum):
            GramVec[i,0]=poly_kernelFun(Feature[i],queryPt,kerPow)
    return GramVec

def get_pred(Feature,Label, kerPow):
    sampNum=len(Label)
    I_N=np.eye(sampNum)
    pred=np.zeros(sampNum)
    if kerPow==0:
        GramMtr=Gau_Gram_Mtr(Feature, Feature)
        for i in range(len(pred)):
            GramVec=Gram_Vec(Feature, Feature[i,:], 0)
            pred[i]=np.dot(Label.T,np.dot(np.linalg.inv(I_N+GramMtr),GramVec))
    else:
        GramMtr=Poly_Gram_Mtr(Feature, Feature, kerPow)
        for i in range(len(pred)):
            GramVec=Gram_Vec(Feature, Feature[i,:], kerPow)
            pred[i]=np.dot(Label.T,np.dot(np.linalg.inv(I_N+GramMtr),GramVec))
    return pred

def get_RMSE(pred, Label):
    return np.sqrt(((pred - Label) ** 2).mean())

RMSE_lst=[]
for i in range(2,5):
    pred=get_pred(trF,trL_nonNorm,i)
    RMSE=get_RMSE(pred, trL_nonNorm)
    RMSE_lst.append(RMSE)

pred=get_pred(trF,trL_nonNorm,0)
RMSE=get_RMSE(pred, trL_nonNorm)
RMSE_lst.append(RMSE)

print RMSE_lst