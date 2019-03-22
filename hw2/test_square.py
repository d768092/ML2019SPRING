import numpy as np
import sys
import csv
data=np.genfromtxt(sys.argv[1],delimiter=',',skip_header=1)
data=np.delete(data,1,axis=1)
cmax=np.load('wmax.npy')
cmin=np.load('wmin.npy')
test=(data-cmin)/(cmax-cmin)
test2=np.power(test,2)
test=np.hstack((np.ones((test.shape[0],1)),test,test2))
w=np.load('model_square.npy')
y=test.dot(w)
one=np.ones([y.shape[0],y.shape[1]])
y=one/(one+np.exp(-y))

with open(sys.argv[2],'w+',newline='') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerow(['id','label'])
    for i in range(y.shape[0]):
        writer.writerow([i+1,int(round(y[i][0]))])
