# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import jieba
import csv
import sys
from gensim.models import Word2Vec
from keras.layers import Dense, GRU, Dropout, Reshape, BatchNormalization, Flatten, Embedding
from keras.utils import to_categorical
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

jieba.set_dictionary(sys.argv[3])

y_train=pd.read_csv(sys.argv[2])['label'].values
y_train = to_categorical(y_train, num_classes=2)
x_train=pd.read_csv(sys.argv[1])['comment'].values

def clearspace(words):
    no_space=[]
    for word in words:
        if not word.isspace():
            no_space.append(word)
    return no_space

train_seg=[]
for sentence in x_train:
    words = clearspace(jieba.lcut(sentence))
    train_seg.append(words)


length = 300
size = 100
w2v = Word2Vec.load('word2vec.model')
embedding_matrix = np.zeros((len(w2v.wv.vocab.items())+1,w2v.vector_size))
for i in range(len(w2v.wv.vocab.items())):
    embedding_matrix[i+1]=w2v.wv[w2v.wv.index2word[i]]

x_train = []
for words in train_seg:
    line_vec=[]
    l=len(words)
    for i in range(length):
        try:
            line_vec.append(w2v.wv.vocab.get(words[i%l]).index+1)
        except:
            line_vec.append(0)
    x_train.append(line_vec)
x_train = np.array(x_train)

model = Sequential()
model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], trainable=True))
model.add(GRU(units=76, return_sequences=True))
model.add(GRU(units=80))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint = ModelCheckpoint('GRU-{epoch:d}.model', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(x_train, y_train, epochs=5, validation_split=0.2, batch_size=128, callbacks=callbacks_list)
model.save('GRU.model')
