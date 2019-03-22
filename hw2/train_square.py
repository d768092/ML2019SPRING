import numpy as np
import sys
data=np.genfromtxt(sys.argv[1],delimiter=',',skip_header=1)
data=np.delete(data,1,axis=1)
data_y=np.genfromtxt(sys.argv[2],skip_header=1).reshape(-1,1)
batch=100
batch_size=data.shape[0]//int(batch*1.1)
epoch=10000
data_max=np.max(data,axis=0).reshape(1,-1)
data_min=np.min(data,axis=0).reshape(1,-1)
data=(data-data_min)/(data_max-data_min)
data2=np.power(data,2)
data=np.hstack((np.ones((data.shape[0],1)),data,data2))
train=[]
train_t=[]
train_y=[]
for i in range(batch):
    train.append(data[batch_size*i:batch_size*(i+1)])
    train_t.append(data[batch_size*i:batch_size*(i+1)].transpose())
    train_y.append(data_y[batch_size*i:batch_size*(i+1)].reshape(-1,1))
valid=data[batch_size*batch:]
valid_y=data_y[batch_size*batch:]
w=np.zeros([data.shape[1],1])
eps=np.full([data.shape[1],1],0.001)
eta=0.0001
beta=0.9
prev_b=1
prev_m=np.zeros([data.shape[1],1])
prev_v=np.zeros([data.shape[1],1])
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

for i in range(epoch):
    if i%100==0:
        print(i,test(data,data_y),test(valid,valid_y))
    for j in range(batch): 
        grad=-(train_t[j].dot(train_y[j]-sigmo(train[j].dot(w))))
        prev_m=beta*prev_m+(1-beta)*grad
        prev_v=beta*prev_v+(1-beta)*(grad**2)
        prev_b*=beta
        m=prev_m/(1-prev_b)
        v=prev_v/(1-prev_b)
        w-=eta*m/(np.sqrt(v)+eps)

print(w)
print(test(valid,valid_y))
np.save('wmax',data_max)
np.save('wmin',data_min)
np.save('model_square',w)
