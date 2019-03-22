import numpy as np
import sys
data=np.genfromtxt(sys.argv[1],delimiter=',',skip_header=1)
data_y=np.genfromtxt(sys.argv[2],skip_header=1)
data_0=[]
data_1=[]
for i in range(data_y.shape[0]):
    if data_y[i]==0:
        data_0.append(data[i])
    elif data_y[i]==1:
        data_1.append(data[i])
data_0=np.array(data_0)
data_1=np.array(data_1)
mean_0=np.mean(data_0,axis=0)
mean_1=np.mean(data_1,axis=0)
cov=np.zeros((data.shape[1],data.shape[1]))
for i in range(data_0.shape[0]):
    cov+=(np.transpose([data_0[i]-mean_0]).dot([data_0[i]-mean_0]))
for i in range(data_1.shape[0]):
    cov+=(np.transpose([data_1[i]-mean_1]).dot([data_1[i]-mean_1]))
cov/=(data_0.shape[0]+data_1.shape[0])
invcov=np.linalg.pinv(cov)
w=(mean_1-mean_0).dot(invcov)
b=(-0.5)*mean_1.dot(invcov).dot(mean_1)\
+0.5*mean_0.dot(invcov).dot(mean_0)+np.log(data_1.shape[0]/data_0.shape[0])
model=np.vstack(([[b]],np.transpose([w])))
print(model)
np.save('generate',model)
