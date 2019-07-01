import numpy as np
import json
import jieba
from gensim.models.word2vec import Word2Vec
from collections import Counter

jieba.set_dictionary("../dict.txt.big")
data = np.load('../content_jieba.npy', allow_pickle = True)
title = np.delete(np.delete(np.genfromtxt("../QS_1.csv", delimiter = ',', dtype = np.str), 0, 0), 0, 1)

query_cnt = Counter()
word_vec = Word2Vec.load("../word2vec.model")

with open("../my_inverted_file") as f:
	invert_file = json.load(f)

for i in title:
	print(i[0])
	query_words = jieba.lcut(i[0], cut_all = False)

	similar = [i[0] for i in word_vec.most_similar(positive = query_words, topn = 3)]
	
	query_cnt.update(query_words)
	query_cnt.update(similar)

dic = {}

for i in range(data.shape[0]):
	dic["news_%06d"%(i + 1)] = 0

cnt = 0
for (word, count) in query_cnt.items():
	print(cnt, end = "\r")
	for i in range(data.shape[0]):
		for j in range(len(data[i])):
			if word == data[i][j]:
				dic["news_%06d"%(i + 1)] += 1
	cnt += 1
with open("../count_total", "w") as output:
	json.dump(dic, output)
