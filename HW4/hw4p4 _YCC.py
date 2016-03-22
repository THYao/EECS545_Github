from __future__ import division
from scipy.ndimage import imread
import numpy as np
from matplotlib import pyplot as plt

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

#The function that can get the labels of each data point
def getlabels(X,centroid):
	labels = np.zeros(X.shape[0])
	for i in range(X.shape[0]):
		dist = np.sum((centroid-X[i])**2,axis=1)
		labels[i] = np.argmin(dist)
	return labels

#The function that can get the centroids of each cluster
def getcentroids(X,label,k):
	centroids = np.zeros((k,X.shape[1]))
	for i in range(k):
		centroids[i,:] = np.mean(X[np.where(label==i)[0],:],axis=0)
	return centroids

#The function that can simulate the k-means process
def k_means(X,k):
	init_cent_label = np.random.choice(X.shape[0],k,replace=False)
	init_centroid = X[init_cent_label,:]
	count = 0
	centroids = init_centroid
	obj_vec = []
	while True:
		label = getlabels(X,centroids)
		new_centroids = getcentroids(X,label,k)
		count += 1
		obj,q_X = obj_fun(X,label,new_centroids)
		obj_vec.append(obj)
		if np.array_equal(new_centroids,centroids):
			break
		else:
			centroids = new_centroids
	return new_centroids,count,obj_vec,q_X

#The function that can return the value of objective function 
def obj_fun(X,label,centroid):
	l = len(label)
	quantized = np.zeros(X.shape)
	for i in range(l):
		quantized[i,:] = centroid[label[i],:]
	obj_value = np.sum((X-quantized)**2)
	return obj_value,quantized

_,count,obj_vec,q_mandrill = k_means(X,k)
iteration_count = np.arange(1,count+1)

#plot of the k-means objective function value versus iteration number

plt.plot(iteration_count,obj_vec,'-')
plt.show()

comp_mandrill = np.zeros(mandrill.shape)
for i in range(N//M):
    for j in range(N//M):
        comp_mandrill[i*M:(i+1)*M,j*M:(j+1)*M,:] = q_mandrill[i*N//M+j,:].reshape(M,M,3)

#compressed mandrill
plt.imshow(comp_mandrill/255)
plt.show()

#original mandrill 
plt.imshow(mandrill/255)
plt.show()

#difference of the two images
plt.imshow((comp_mandrill - mandrill + 128)/255)
plt.show()

#compression ratio
cr = np.log2(k)/(24*M**2)+k*M**2/N**2
print(cr)
#relative mean absolute error of the compressed image
rmaa = np.sum(abs(comp_mandrill-mandrill))*(1/(3*N**2))/255
print(rmaa)

