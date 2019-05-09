# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import sys
import jieba
from gensim.models import Word2Vec

jieba.set_dictionary(sys.argv[3])

x_train=pd.read_csv(sys.argv[1])['comment'].values
x_test=pd.read_csv(sys.argv[2])['comment'].values
x_data=np.concatenate((x_train,x_test),axis=0)

def clearspace(words):
    no_space=[]
    for word in words:
        if not word.isspace():
            no_space.append(word)
    return no_space

data_seg=[]
for sentence in x_data:
    words = clearspace(jieba.lcut(sentence))
    data_seg.append(words)

model = Word2Vec(data_seg, size=300, iter=15, window=5, min_count=3)
model.save('word2vec.model')
