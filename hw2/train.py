import numpy as np
import sys
data=np.genfromtxt(sys.argv[1],delimiter=',',skip_header=1)
data_y=np.genfromtxt(sys.argv[2],skip_header=1)
train=np.ones([data.shape[0],1])
train=np.hstack((train,data))
train_t=np.transpose(train)
train_y=np.array([data_y]).transpose()
w=np.zeros([train.shape[1],1])
eps=np.full([train.shape[1],1],0.001)
eta=0.0001
t=100000
beta=0.9
prev_b=1
prev_m=np.zeros([train.shape[1],1])
prev_v=np.zeros([train.shape[1],1])
def sigmo(npar):
    one=np.ones([npar.shape[0],npar.shape[1]])
    npar=one/(one+np.exp(-npar))
    return npar
def test(test_data,test_ans):
    y=sigmo(test_data.dot(w))
    rate=0
    for i in range(y.shape[0]):
        predict=int(round(y[i][0]))
        rate+=(test_ans[i][0]*predict+(1-test_ans[i][0])*(1-predict))
    rate/=y.shape[0]
    return rate

for i in range(t):
    if i%1000==0:
        print(i,test(train,train_y))
    grad=-(train_t.dot(train_y-sigmo(train.dot(w))))
    prev_m=beta*prev_m+(1-beta)*grad
    prev_v=beta*prev_v+(1-beta)*(grad**2)
    prev_b*=beta
    m=prev_m/(1-prev_b)
    v=prev_v/(1-prev_b)
    w-=eta*m/(np.sqrt(v)+eps)
print(w)
print(test(train,train_y))
np.save('model',w)
