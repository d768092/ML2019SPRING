import pandas as pd
import numpy as np
import sys
import csv
import os
import tensorflow as tf
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, MaxPooling2D, BatchNormalization, DepthwiseConv2D, Dropout, GlobalAveragePooling2D, Reshape, Flatten, Dense
import keras.backend as K

model=Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(DepthwiseConv2D((3,3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Conv2D(32, (1,1)))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(DepthwiseConv2D((3,3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Conv2D(32, (1,1)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(2,2))

model.add(DepthwiseConv2D((3,3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Conv2D(48, (1,1)))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(DepthwiseConv2D((3,3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Conv2D(48, (1,1)))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(DepthwiseConv2D((3,3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Conv2D(64, (1,1)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(DepthwiseConv2D((3,3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Conv2D(64, (1,1)))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(DepthwiseConv2D((3,3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Conv2D(96, (1,1)))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(DepthwiseConv2D((3,3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Conv2D(96, (1,1)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(units=7,activation='softmax'))
w = np.load('model_aug.npy', allow_pickle=True)
model.set_weights(w)

data=pd.read_csv(sys.argv[1])
test=data['feature'].values
test=np.array([[float(i)/255 for i in feature.split()] for feature in test]).reshape(-1,48,48,1)
predictions=model.predict_classes(test)
with open(sys.argv[2],'w+',newline='') as output:
    writer=csv.writer(output)
    writer.writerow(['id','label'])
    for i in range(len(predictions)):
        writer.writerow([i,predictions[i]])
print(predictions)
