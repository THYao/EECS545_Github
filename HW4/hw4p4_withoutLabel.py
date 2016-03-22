from __future__ import division
from scipy.ndimage import imread
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA

# Load the mandrill image as an NxNx3 array. Values range from 0.0 to 255.0.
mandrill = imread('mandrill.png', mode='RGB').astype(float)
N = int(mandrill.shape[0])

M = 2
k = 64

# Store each MxM block of the image as a row vector of X
X = np.zeros((N**2//M**2, 3*M**2))
for i in range(N//M):
    for j in range(N//M):
        X[i*N//M+j,:] = mandrill[i*M:(i+1)*M,j*M:(j+1)*M,:].reshape(3*M**2)


# TODO: Implement k-means and cluster the rows of X, then reconstruct the
# compressed image using the cluster center for each block, as specified in
# the homework description.


# To show a color image using matplotlib, you have to restrict the color
# color intensity values to between 0.0 and 1.0. For example,

def AssignLabel(allDataLabel,allCentroid):
    sampleNum=len(allDataLabel)
    labelIndex=allDataLabel.shape[1]-1
    dataVector=allDataLabel[:,0:labelIndex]
    for i in range(sampleNum):
        idx = (LA.norm(dataVector[i]-allCentroid,axis=1)).argmin()
        allDataLabel[i][labelIndex]=idx
    return allDataLabel

def getCentroid(allDatawithLabel,numOfGr):
    labelIndex=np.shape(allDatawithLabel)[1]-1
    dataCentroid=[]
    for i in range(numOfGr):
        group_i=allDatawithLabel[allDatawithLabel[:,labelIndex]==i]
        centroid_i=np.mean(group_i[:,0:labelIndex],axis=0)
        dataCentroid.append(centroid_i.tolist())
    dataArrayCentroid=np.array(dataCentroid)
    return dataArrayCentroid
    
def initStep(data,numOfGr):
    init_cent_label = np.random.choice(data.shape[0],numOfGr,replace=False)
    centroid_init=data[init_cent_label,:]
    zero=np.zeros(data.shape[0])
    datawithLabel=np.column_stack((data,zero))
    return datawithLabel, centroid_init
    
def checkTerminal(centroidOld,centroidNew):
    cenOldsort=np.sort(centroidOld)
    cenNewsort=np.sort(centroidNew)
    return np.all(cenOldsort==cenNewsort)

def genClustered_data(dataLabel,centroid):
    labelIndex=dataLabel.shape[1]-1
    label=dataLabel[:,labelIndex]
    clusterData=[]
    for i in range(len(label)):
        cIdx=int(label[i])
        dataTobeApp=centroid[cIdx].tolist()
        dataTobeApp.append(cIdx)
        clusterData.append(dataTobeApp)
    clusterData=np.array(clusterData)
    return clusterData

def obj_fn(orgD,cluD):
    value=np.sum((orgD-cluD)**2)
    return value

iteNum=0
obj_list=[]
dataLabel_old,centroid_old=initStep(X,k)
dataNewLabel=AssignLabel(dataLabel_old,centroid_old)     
centroid_New=getCentroid(dataNewLabel,k)
cluD=genClustered_data(dataNewLabel,centroid_New)
obj_value=obj_fn(dataNewLabel,cluD)
obj_list.append(obj_value)

while not checkTerminal(centroid_old,centroid_New):
    iteNum+=1
    centroid_old=centroid_New
    dataLabel_old=dataNewLabel
    dataNewLabel=AssignLabel(dataLabel_old,centroid_old)
    centroid_New=getCentroid(dataNewLabel,k)
    cluD=genClustered_data(dataNewLabel,centroid_New)
    obj_value=obj_fn(dataNewLabel,cluD)
    obj_list.append(obj_value)

print centroid_New

ite=np.arange(iteNum+1)
plt.plot(ite,obj_list,'-')
plt.xlabel("iteration")
plt.ylabel("objective function")
plt.legend()
plt.show()

cluD_mandrill=cluD[:,0:12]
comp_mandrill = np.zeros(mandrill.shape)
for i in range(N//M):
    for j in range(N//M):
        comp_mandrill[i*M:(i+1)*M,j*M:(j+1)*M,:] = cluD_mandrill[i*N//M+j,:].reshape(M,M,3)

plt.imshow(comp_mandrill/255)
plt.show()

plt.imshow(mandrill/255)
plt.show()

plt.imshow((comp_mandrill - mandrill + 128)/255)
plt.show()

CompR = np.log2(k)/(24*M**2)+k*M**2/N**2
print(CompR)

RMAE = np.sum(abs(comp_mandrill-mandrill))*(1/(3*N**2))/255
print(RMAE)