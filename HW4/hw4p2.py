from __future__ import division
import numpy as np
from scipy.special import polygamma, gammaln
from matplotlib import pyplot as plt
# Generate the data according to the specification in the homework description
   
N = 1000
m = 5
alpha = np.array([10, 5, 15, 20, 50])
P = np.random.dirichlet(alpha, N)
P_f=P
print np.shape(P)
# TODO: Implement the Newton-Raphson algorithm for estimating the parameters of
# the Dirichlet distribution given observances (rows of P).
LogData=np.log(P_f)
SuffStat=np.mean(LogData,axis=0)

def LogLike(alphaIn):
    sumAlpha=np.sum(alphaIn)    
    output=N*(gammaln(sumAlpha)-np.sum(gammaln(alphaIn))+np.dot((alphaIn-1),SuffStat))
    return output

def Gradient(alphaIn):
    sumAlpha=np.sum(alphaIn)
    gradient=N*(polygamma(0,sumAlpha)-polygamma(0,alphaIn)+SuffStat)
    return gradient

def QMatrix(alphaIn):
    Q_Matrix=np.diag(-N*polygamma(1,alphaIn))
    return Q_Matrix

def RealNumC(alphaIn):
    ConstC=N*polygamma(1,np.sum(alphaIn))
    return ConstC
    
def Update(alphaIn):
    ConstC=RealNumC(alphaIn)
    oneV=np.ones((m,m))
    QInv=np.diag((1.0/QMatrix(alphaIn)).diagonal())
    GV=Gradient(alphaIn)
    output=np.dot(QInv,GV)-1.0/(1.0/ConstC+np.sum(QInv.diagonal()))*np.dot(np.dot(QInv,oneV),np.dot(QInv,GV))
    return output

alphaInit=np.array([1,1,1,1,1])
logLike_lst=[]
True_lst=[]
TrueLogLike=LogLike(alpha)
alphaOld=alphaInit
logLike_lst.append(LogLike(alphaOld))
True_lst.append(TrueLogLike)
alphaNew=alphaOld-Update(alphaOld)
logLike_lst.append(LogLike(alphaNew))
True_lst.append(TrueLogLike)
count=1

while abs(LogLike(alphaNew)-LogLike(alphaOld))>10**(-4):
    alphaOld=alphaNew    
    alphaNew=alphaOld-Update(alphaOld)
    print LogLike(alphaNew), LogLike(alpha)
    print alphaNew
    logLike_lst.append(LogLike(alphaNew))
    True_lst.append(TrueLogLike)
    count+=1
else:
    print alphaNew

TrueLogLike=LogLike(alpha)

count_lst=np.arange(count+1)
plt.plot(count_lst,logLike_lst,'-',label="MLE")
plt.plot(count_lst,True_lst,'r-',label="True")
plt.xlabel("iteration")
plt.ylabel("log-likelihood")
plt.legend()
plt.show()