# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import jieba
import csv
import sys
from gensim.models import Word2Vec
from keras.models import load_model

jieba.set_dictionary(sys.argv[2])

x_test=pd.read_csv(sys.argv[1])['comment'].values

def clearspace(words):
    no_space=[]
    for word in words:
        if not word.isspace():
            no_space.append(word)
    return no_space

test_seg=[]
for sentence in x_test:
    words = clearspace(jieba.lcut(sentence))
    test_seg.append(words)

length = 100
size = 300
w2v = Word2Vec.load('word2vec.model')

x_test = []
comma = w2v.wv['，']
for words in test_seg:
    line_vec=[]
    l = len(words)
    for i in range(length):
        try:
            line_vec.append(w2v.wv.vocab.get(words[i%l]).index+1)
            #line_vec.append(w2v.wv[words[i%l]])
        except:
            line_vec.append(0)
            #line_vec.append(comma)
    x_test.append(line_vec)
x_test=np.array(x_test)

model1=load_model('LSTM2-2.model')
prediction1=model1.predict_classes(x_test)
model2=load_model('conv.model')
prediction2=model2.predict_classes(x_test)
model3=load_model('GRU-2.model')
prediction3=model3.predict_classes(x_test)
predictions=np.round((prediction1+prediction2+prediction3)/3).astype(np.int32)

with open(sys.argv[3],'w+',newline='') as output:
    writer=csv.writer(output)
    writer.writerow(['id','label'])
    for i in range(len(predictions)):
        writer.writerow([i,predictions[i]])
print(predictions)
