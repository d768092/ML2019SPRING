import numpy as np
import pandas as pd
import sys
import csv
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, MaxPooling2D, BatchNormalization, DepthwiseConv2D, Dropout, GlobalAveragePooling2D, Reshape, Flatten, Dense
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

data=pd.read_csv(sys.argv[1])
train_y=data['label'].values
train_x=data['feature'].values
train_x=np.array([[float(i)/255 for i in feature.split()] for feature in train_x]).reshape(-1,48,48,1)
num_classes=7
train_y=to_categorical(train_y,num_classes)
train_num = int(train_x.shape[0]*0.8)
valid_x=train_x[train_num:]
valid_y=train_y[train_num:]
train_x=train_x[:train_num]
train_y=train_y[:train_num]

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
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

reduce_lr = ReduceLROnPlateau()
checkpoint = ModelCheckpoint('model_aug-{epoch:d}.check', monitor='val_acc', 
        verbose=1, save_best_only=True, mode='max', period=20)
callbacks_list = [reduce_lr, checkpoint]

datagen=ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)
datagen.fit(train_x)
model.fit_generator(datagen.flow(train_x,train_y,batch_size=32),
    steps_per_epoch=train_x.shape[0]//32, validation_data=(valid_x,valid_y), 
    epochs=200, callbacks=callbacks_list)
w=model.get_weights()
w = [i.astype('float16') for i in w]
np.save('model_aug',w)
#model.save('model_aug.h5')
