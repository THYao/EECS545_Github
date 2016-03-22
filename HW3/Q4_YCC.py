import numpy as np

traindt_raw = np.loadtxt("steel_composition_train.csv",delimiter=',',skiprows=1,usecols=(1,2,3,4,5,6,7,8,9))
rowcount_tr = traindt_raw.shape[0]
one_tr = np.ones((rowcount_tr,1))
traindt = np.concatenate((one_tr,traindt_raw[:,0:8]),axis=1)
traindt_tg = traindt_raw[:,8]

def normalization(array):
	mean_array = np.mean(array,axis=0)
	sd_array = np.std(array,axis=0)
	norm_array = np.zeros(array.shape)
	for i in range(array.shape[1]):
		for j in range(array.shape[0]):
			norm_array[j,i]=(array[j,i]-mean_array[i])/sd_array[i]
	return norm_array

traindt = normalization(traindt[:,1:])


def kernel1(u,v):
    return (sum(u*v)+1)**2
def kernel2(u,v):
    return (sum(u*v)+1)**3
def kernel3(u,v):
    return (sum(u*v)+1)**4
def kernel4(u,v):
    return np.exp(-np.linalg.norm(u-v)**2/2.0)

K_gram1 = np.zeros((rowcount_tr,rowcount_tr))
K_gram2 = np.zeros((rowcount_tr,rowcount_tr))
K_gram3 = np.zeros((rowcount_tr,rowcount_tr))
K_gram4 = np.zeros((rowcount_tr,rowcount_tr))
for i in range(rowcount_tr):
    for j in range(rowcount_tr):
        K_gram1[i,j] = kernel1(traindt[i,:],traindt[j,:])
        K_gram2[i,j] = kernel2(traindt[i,:],traindt[j,:])   
        K_gram3[i,j] = kernel3(traindt[i,:],traindt[j,:])
        K_gram4[i,j] = kernel4(traindt[i,:],traindt[j,:])

t_pred1 = np.zeros(rowcount_tr)
t_pred2 = np.zeros(rowcount_tr)
t_pred3 = np.zeros(rowcount_tr)
t_pred4 = np.zeros(rowcount_tr)

for row in range(rowcount_tr):
    k_vec1 = np.zeros(rowcount_tr)
    k_vec2 = np.zeros(rowcount_tr)
    k_vec3 = np.zeros(rowcount_tr)
    k_vec4 = np.zeros(rowcount_tr)
    for j in range(rowcount_tr):
        k_vec1[j] = kernel1(traindt[row,:],traindt[j,:])
        k_vec2[j] = kernel2(traindt[row,:],traindt[j,:])
        k_vec3[j] = kernel3(traindt[row,:],traindt[j,:])
        k_vec4[j] = kernel4(traindt[row,:],traindt[j,:])
    t_pred1[row] = traindt_tg.T.dot(np.linalg.inv(np.eye(rowcount_tr)+K_gram1)).dot(k_vec1)
    t_pred2[row] = traindt_tg.T.dot(np.linalg.inv(np.eye(rowcount_tr)+K_gram2)).dot(k_vec2)
    t_pred3[row] = traindt_tg.T.dot(np.linalg.inv(np.eye(rowcount_tr)+K_gram3)).dot(k_vec3)
    t_pred4[row] = traindt_tg.T.dot(np.linalg.inv(np.eye(rowcount_tr)+K_gram4)).dot(k_vec4)

def RMSE(fit,real):
    diff = fit-real
    sum = 0
    for n in range(len(diff)):
        sum = sum+diff[n]**2
    return (np.sqrt(sum/len(fit)))
    
rmse = np.zeros(4)
rmse[0]=RMSE(t_pred1,traindt_tg)
rmse[1]=RMSE(t_pred2,traindt_tg)
rmse[2]=RMSE(t_pred3,traindt_tg)
rmse[3]=RMSE(t_pred4,traindt_tg)
print(rmse)

