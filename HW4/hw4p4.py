from __future__ import division
from scipy.ndimage import imread
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean

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
plt.imshow(mandrill/255)
plt.show()

def AssignLabel(singleDataPt,allCentroid):
    labelIndex=len(singleDataPt)-1  
    dataLabel=singleDataPt[labelIndex]
    dataVector=singleDataPt[:labelIndex]
    for i in range(len(allCentroid)):
        centroidLabel_i=allCentroid[i][labelIndex]
        centroidVector_i=allCentroid[i][:labelIndex]
        if i ==0:
            leastDist=euclidean(dataVector,centroidVector_i)
            dataLabel=centroidLabel_i
        else:
            if euclidean(dataVector,centroidVector_i)<leastDist:
                dataLabel=centroidLabel_i   
    return dataLabel

def getCentroid(allDatawithLabel,i):
    labelIndex=np.shape(allDatawithLabel)[1]-1
    dataVector=allDatawithLabel[:,0:labelIndex]
    dataCentroid=np.mean(dataVector,axis=0)
    data=np.hstack((dataCentroid,i))
    return dataCentroid
    
def initStep(data,numOfGr):
    init_cent_label = np.random.choice(data.shape[0],numOfGr,replace=False)
    centroidVector=data[init_cent_label,:]
    centroidLabel=np.arange(numOfGr)
    centroid=np.column_stack((centroidVector,centroidLabel))
    zero=np.zeros(data.shape[0])
    datawithLabel=np.column_stack((data,zero))
    return datawithLabel, centroid
    

        
    
    