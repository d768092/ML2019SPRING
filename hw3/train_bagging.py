import numpy as np
import pandas as pd
import sys
from keras import backend as K
from keras.engine.network import Network
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from collections import Counter

data=pd.read_csv(sys.argv[1])
data_y=data['label'].values
data_x=data['feature'].values
data_x=np.array([[float(i)/255 for i in feature.split()] for feature in data_x]).reshape(-1,48,48,1)
num_classes=7
data_y=to_categorical(data_y,num_classes)
num_train=data_x.shape[0]

x_train=[]
y_train=[]
for i in range(4):
    x_train.append(data_x[num_train*i//4:num_train*(i+1)//4])
    y_train.append(data_y[num_train*i//4:num_train*(i+1)//4])

model=Sequential()
model.add(Conv2D(64,3,3,input_shape=(48,48,1),activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,3,3,activation='relu'))
model.add(Conv2D(128,3,3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Conv2D(256,3,3,activation='relu'))
model.add(Conv2D(256,3,3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(units=1024,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=7,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
Wsave=model.get_weights()

datagen=ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

train_bag=[(0,2),(0,3),(1,2),(1,3),(0,1),(2,3)]
valid_bag=[(1,3),(1,2),(0,3),(0,2),(2,3),(0,1)]
bags=int(sys.argv[3])
scores=[]
predictions=[ [] for i in range(num_train) ]
for i in range(bags):
    train=np.vstack((x_train[train_bag[i][0]],x_train[train_bag[i][1]]))
    train_y=np.vstack((y_train[train_bag[i][0]],y_train[train_bag[i][1]]))
    valid=np.vstack((x_train[valid_bag[i][0]],x_train[valid_bag[i][1]]))
    valid_y=np.vstack((y_train[valid_bag[i][0]],y_train[valid_bag[i][1]]))
    model.fit(train, train_y, batch_size=256, epochs=100,)
    
    datagen.fit(train)
    model.fit_generator(datagen.flow(train,train_y,batch_size=32),
    steps_per_epoch=train.shape[0]//32, epochs=500)
    
    score=model.evaluate(valid,valid_y)
    print(score)
    scores.append(score)
    prediction=model.predict_classes(valid)
    now=0
    for j in range(2):
        for k in range(num_train*valid_bag[i][j]//4,num_train*(valid_bag[i][j]+1)//4):
            predictions[k].append(prediction[now])
            now+=1

    model.save(sys.argv[2]+'_'+str(i))
    model.set_weights(Wsave)

for i in range(bags):
    print("model",i,": ",scores[i])

correct=0
for i in range(num_train):
    ans=Counter(predictions[i]).most_common(1)[0][0]
    if data_y[i][ans]==1:
        correct+=1
print(correct/num_train)
