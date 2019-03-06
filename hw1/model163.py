import numpy as np
import sys
data=np.genfromtxt(sys.argv[1],delimiter=',',skip_header=1)
for i in range(0,data.shape[0],18):
    tmpnumdata=data[i:i+18,3:]
    if i==0:
        numdata=tmpnumdata
    else:
        numdata=np.hstack((numdata,tmpnumdata))
numdata[np.isnan(numdata)]=0
for i in range(numdata.shape[1]-9):
    tmptrain=[1]
    tmpy=[numdata[9,i+9]]
    for j in range(numdata.shape[0]):
        tmptrain=np.hstack((tmptrain,numdata[j,i:i+9]))
    if i==0:
        train=tmptrain
        train_y=tmpy
    else:
        train=np.vstack((train,tmptrain))
        train_y=np.hstack((train_y,tmpy))
train_t=np.transpose(train)
w=np.zeros([1,163])
eta=0.1
t=100000
prev_grad=0
for i in range(t):
    grad=(-2)*(np.dot(train_y,train)-np.dot(w,np.dot(train_t,train)))
    prev_grad+=grad**2
    ada=np.sqrt(prev_grad)
    w-=eta*grad/ada
np.save('predict163',w)
