from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
# Generate the data according to the specification in the homework description

N = 500
x = np.random.rand(N)

pi0 = np.array([0.7, 0.3])
w0 = np.array([-2, 1])
b0 = np.array([0.5, -0.5])
sigma0 = np.array([.4, .3])

y = np.zeros_like(x)
for i in range(N):
    k = 0 if np.random.rand() < pi0[0] else 1
    y[i] = w0[k]*x[i] + b0[k] + np.random.randn()*sigma0[k]
std_y=np.std(y,ddof=1)

# TODO: Implement the EM algorithm for Mixed Linear Regression based on observed
# x and y values.

# Here's the data plotted
plt.scatter(x, y, c='r', marker='x')
plt.show()

def LogLike(y_D,x_D,pi_old,w_old, b_old, sigma_old):
    output=0
    for n in range(N):
        for i in range(len(pi_old)):
            output+=pi_old[i]*norm.pdf(y_D[n],w_old[i]*x_D[n]+b_old[i],sigma_old[i])
    return output
        

def r_ik(x_i,y_i,pi_old,w_old, b_old, sigma_old):    
    numer=pi_old[0]*norm.pdf(y_i,w_old[0]*x_i+b_old[0],sigma_old[0])
    deno=0
    for i in range(len(pi_old)):
        deno+=pi_old[i]*norm.pdf(y_i,w_old[i]*x_i+b_old[i],sigma_old[i])
    r_i0=numer/deno
    r_i1=1-r_i0
    output=np.vstack((r_i0,r_i1))
    return output

def pi_update(r_ikOld):
    piNew=[]
    for k in range(len(pi0)):
        piNew.append(np.sum(r_ikOld[k]/500))
    return piNew

def beta_update(x_D,y_D,r_ikOld):
    bNew=[]
    wNew=[]
    One=np.ones(N)
    S_D=np.vstack((One,x)).T
    for k in range(2):
        R_k=np.diag(r_ikOld[k])
        beta_k=np.dot(np.linalg.pinv(np.dot(np.dot(S_D.T,R_k),S_D)),np.dot(S_D.T,np.dot(R_k,y_D)))
        bNew.append(beta_k[0])
        wNew.append(beta_k[1])
    return bNew, wNew

def sigma_update(x_D,y_D,w_old,b_old,r_ikOld):  
    sigmaNew=[]    
    for k in range(2):
        deno=np.sum(r_ikOld[k])
        nume=np.sum(r_ikOld[k]*(y_D-w_old[k]*x_D-b_old[k])**2)
        sigmaNew.append((nume/deno)**0.5)
    return sigmaNew
    
pi_init=np.array([0.5,0.5])
w_init=np.array([1,-1])
b_init=np.array([0,0])
sigma_init=np.array([std_y,std_y])
iteCount=0
LogLike_lst=[]
piOld=pi_init
wOld=w_init
bOld=b_init
sigmaOld=sigma_init
LogLike_lst.append(LogLike(y,x,piOld,wOld, bOld, sigmaOld))
r_ikOld=r_ik(x,y,piOld,wOld,bOld,sigmaOld)

piNew=pi_update(r_ikOld)
bNew, wNew=beta_update(x,y,r_ikOld)
sigmaNew=sigma_update(x,y,wNew,bNew,r_ikOld)
iteCount+=1
LogLike_lst.append(LogLike(y,x,piNew,wNew, bNew, sigmaNew))

while abs(LogLike(y,x,piOld,wOld, bOld, sigmaOld)-LogLike(y,x,piNew,wNew, bNew, sigmaNew))>10**(-4):
    piOld=piNew
    wOld=wNew
    bOld=bNew
    sigmaOld=sigmaNew
    r_ikOld=r_ik(x,y,piOld,wOld,bOld,sigmaOld)
    piNew=pi_update(r_ikOld)
    bNew, wNew=beta_update(x,y,r_ikOld)
    sigmaNew=sigma_update(x,y,wNew,bNew,r_ikOld)
    LogLike_lst.append(LogLike(y,x,piNew,wNew, bNew, sigmaNew))
    iteCount+=1    
    
print piNew
print bNew
print wNew
print sigmaNew

count_lst=np.arange(iteCount+1)
plt.plot(count_lst,LogLike_lst,'-')
plt.xlabel("iteration")
plt.ylabel("log-likelihood")
plt.legend()
plt.show()

Line1 = wNew[0]*x+bNew[0]
Line2 = wNew[1]*x+bNew[1]
plt.scatter(x, y, c='r', marker='x')
plt.plot(x,Line1,'g-')
plt.plot(x,Line2,'b-')
plt.show()