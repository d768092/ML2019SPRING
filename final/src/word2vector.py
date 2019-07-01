from gensim.models.word2vec import Word2Vec
import numpy as np
import sys

#load data
word_data = np.load("../content_and_title_jieba.npy", allow_pickle = True).tolist()

#train word2vector model
model = Word2Vec(word_data, size = 200, iter = 50, workers = 16)

#save model
model.save('../word2vec.model')
