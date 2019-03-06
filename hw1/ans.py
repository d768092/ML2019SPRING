import numpy as np
import sys
import csv
data=np.genfromtxt(sys.argv[1],delimiter=',')
numdata=data[:,2:]
numdata[np.isnan(numdata)]=0
for i in range(numdata.shape[0]):
    for j in range(numdata.shape[1]):
        if(numdata[i][j]<0):
            numdata[i][j]=0
w=np.load(sys.argv[2])
for i in range(0,numdata.shape[0],18):
    tmptest=[1]
    for j in range(18):
        tmptest=np.hstack((tmptest,numdata[i+j]))
    if i==0:
        test=tmptest
    else:
        test=np.vstack((test,tmptest))
test_t=np.transpose(test)
y=np.dot(w,test_t)
with open(sys.argv[3],'w+',newline='') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerow(['id','value'])
    for i in range(y.shape[1]):
        writer.writerow(['id_'+str(i),y[0][i]])
