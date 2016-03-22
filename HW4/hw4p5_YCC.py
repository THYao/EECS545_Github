from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

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


# TODO: Implement the EM algorithm for Mixed Linear Regression based on observed
# x and y values.
def normal_den(X,Y,beta,sigma,n_ind,k_ind):
	density = np.exp(-(Y[n_ind]-X[n_ind,:].dot(beta[:,k_ind]))**2/(2*sigma[k_ind]**2))/np.sqrt(2*np.pi*sigma[k_ind]**2)
	return density

def responsibility(pi,X,Y,beta,sigma,n_ind,k_ind):
    temp = np.zeros(2)
    for k in range(2):
        temp[k] = pi[k]*normal_den(X,Y,beta,sigma,n_ind,k)
    r_nk = temp[k_ind]/sum(temp)
    return r_nk

def log_likelihood(X,Y,pi,beta,sigma,N=500,K=2):
	logllh = 0
	for n in range(N):
		temp = 0
		for k in range(K):
			temp += pi[k]*normal_den(X,Y,beta,sigma,n,k)
		logllh += np.log(temp)
	return logllh 

pi = np.array([0.5,0.5])
w = np.array([1.,-1.])
b = np.array([0.,0.])
std = np.std(y)
sigma = np.array([std,std])
one_matrix = np.ones((N,1))
x_reshape = x.reshape((N,1))
x_mat = np.concatenate((one_matrix,x_reshape),axis=1)
y_mat = y.reshape((N,1))
beta_matrix = np.array([b[0],b[1],w[0],w[1]]).reshape(2,2)
r_nk = np.zeros((N,2))
logllh = [0]
count = 0

while True:
	#expectation
	for k in range(2):	
		for n in range(N):
			r_nk[n,k] = responsibility(pi,x_mat,y_mat,beta_matrix,sigma,n,k)
	#maximization
	for k in range(2):
		r_mat = np.diag(r_nk[:,k])
		pi[k] = sum(r_nk[:,k])/N
		beta_temp = np.linalg.inv(x_mat.T.dot(r_mat).dot(x_mat)).dot(x_mat.T).dot(r_mat).dot(y_mat)
		b[k],w[k] = beta_temp[0],beta_temp[1]
		temp = 0
		for n in range(N):
			temp += r_nk[n,k]*(y_mat[n]-w[k]*x[n]-b[k])**2
		sigma[k] = np.sqrt(temp/sum(r_nk[:,k]))
	beta_matrix = np.array([b[0],b[1],w[0],w[1]]).reshape(2,2)
	count += 1
	new_logllh = log_likelihood(x_mat,y_mat,pi,beta_matrix,sigma)
	logllh.append(new_logllh)
	if abs(logllh[count]-logllh[count-1])<10**(-4):
		break

iteration_count = np.arange(count+1)
plt.plot(iteration_count[1:],logllh[1:])
plt.show()
print (count)
print ("pi_1 estimate is ",pi[0],",pi_2 estimate is ",pi[1])
print ("w_1 estimate is ",w[0],",w_2 estimate is ",w[1])
print ("b_1 estimate is ",b[0],",b_2 estimate is ",b[1])
print ("sigma_1 estimate is ",sigma[0],",sigma_2 estimate is ",sigma[1])
# Here's the data plotted
y1 = w[0]*x+b[0]
y2 = w[1]*x+b[1]
plt.scatter(x, y, c='r', marker='x')
plt.plot(x,y1,'g-')
plt.plot(x,y2,'b-')
plt.show()

    