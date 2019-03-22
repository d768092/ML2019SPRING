import numpy as np
import sys
import csv
data=np.genfromtxt(sys.argv[1],delimiter=',',skip_header=1)
test=np.ones([data.shape[0],1])
test=np.hstack((test,data))
w=np.load('model.npy')
y=test.dot(w)
one=np.ones([y.shape[0],y.shape[1]])
y=one/(one+np.exp(-y))


with open(sys.argv[2],'w+',newline='') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerow(['id','label'])
    for i in range(y.shape[0]):
        writer.writerow([i+1,int(round(y[i][0]))])
