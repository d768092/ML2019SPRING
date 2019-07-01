import json
import jieba
import pandas as pd
import numpy as np
import random
import csv
import operator
from gensim.models.word2vec import Word2Vec
from argparse import ArgumentParser
from collections import Counter

parser = ArgumentParser()
parser.add_argument("-i", "--inverted_file", default='../my_inverted_file', dest = "inverted_file", help = "Pass in a .json file.")
parser.add_argument("-q", "--query_file", default='../QS_1.csv', dest = "query_file", help = "Pass in a .csv file.")
parser.add_argument("-c", "--corpus_file", default='../NC_1.csv', dest = "corpus_file", help = "Pass in a .csv file.")
parser.add_argument("-o", "--output_file", default='../sample_output.csv', dest = "output_file", help = "Pass in a .csv file.")
parser.add_argument("-t", "--TD_file", default='../TD.json', dest = "TD", help = "Pass in a .json file.")
parser.add_argument("-w", "--word_vec", default='../word2vec.model', dest = "word_vec", help = "Pass in a .json file.")
parser.add_argument("-C", "--count_file", default='../count_total', dest = "count", help = "Pass in a .json file.")

args = parser.parse_args()

# load inverted file
with open(args.inverted_file) as f:
	invert_file = json.load(f)

with open(args.TD) as f:
	TD = json.load(f)

with open(args.count) as f:
	word_count = json.load(f)

jieba.set_dictionary("../dict.txt.big")

#reverse_word = ['反對', '反對', '反對', '反對', '反對', '反對', '反對', '反對', '反對', '反對', '贊成', '贊成', '贊成', '反對', '贊成', '反對', '反對', '反對', '反對', '贊成']

# read query and news corpus
querys = np.array(pd.read_csv(args.query_file)) # [(query_id, query), (query_id, query) ...]
corpus = np.array(pd.read_csv(args.corpus_file)) # [(news_id, url), (news_id, url) ...]
num_corpus = corpus.shape[0] # used for random sample

# process each query
final_ans = []

word_vec = Word2Vec.load(args.word_vec)
cnt = 0

for (query_id, query) in querys:
	print("query_id: {}".format(query_id))
	
	# counting query term frequency
	query_cnt = Counter()
	query_words = list(jieba.cut(query))

	similar = [i[0] for i in word_vec.most_similar(positive = query_words, topn = 3)]
	print(similar)
	similar_dic = {}
	for i in word_vec.most_similar(positive = query_words, topn = 3):
		similar_dic[i[0]] = i[1]
	#for i in word_vec.most_similar(positive = [reverse_word[cnt]], topn = 5):
	#	similar_dic[i[0]] = -10000
	#similar_dic[reverse_word[cnt]] = -10000
	for i in query_words:
		if invert_file[i]['idf'] < 1.5:
			similar_dic[i] = 0
		else:
			similar_dic[i] = 1
	
	query_cnt.update(query_words)
	query_cnt.update(similar)
	#query_cnt.update([reverse_word[cnt]])

	# calculate scores by tf-idf
	document_scores = dict() # record candidate document and its scores
	for (word, count) in query_cnt.items():
		if word in invert_file:
			query_tf = count
			idf = invert_file[word]['idf']
			for document_count_dict in invert_file[word]['docs']:
				for doc, doc_tf in document_count_dict.items():
					if doc in document_scores:
						document_scores[doc] += query_tf * idf * (doc_tf / word_count[doc]) ** (0.5) * idf * similar_dic[word] / (idf ** 0.5)
					else:
						document_scores[doc] = query_tf * idf * (doc_tf / word_count[doc]) ** (0.5) * idf * similar_dic[word] / (idf ** 0.5)
	# sort the document score pair by the score
	sorted_document_scores = sorted(document_scores.items(), key=operator.itemgetter(1), reverse=True)
	
	if query in TD:
		final_ans.append([i[0] for i in TD[query]])
	else:
		final_ans.append([])

	# record the answer of this query to final_ans
	if len(sorted_document_scores) >= 300 - len(final_ans[-1]):
		now = len(final_ans[-1])
		temp = [doc_score_tuple[0] for doc_score_tuple in sorted_document_scores]
		for i in temp:
			if i in final_ans[-1]:
				continue
			final_ans[-1].append(i)
			now += 1
			if now == 300:
				break
	else: # if candidate documents less than 300, random sample some documents that are not in candidate list
		documents_set  = set([doc_score_tuple[0] for doc_score_tuple in sorted_document_scores])
		sample_pool = ['news_%06d'%news_id for news_id in range(1, num_corpus+1) if 'news_%06d'%news_id not in documents_set and 'news_%06d'%news_id not in final_ans[-1]]
		sample_ans = random.sample(sample_pool, 300-count-len(final_ans[-1]))
		sorted_document_scores.extend(sample_ans)
		temp = [doc_score_tuple[0] for doc_score_tuple in sorted_document_scores]
		for i in temp:
			final_ans[-1].append(i)
	cnt += 1
	
# write answer to csv file
with open(args.output_file, 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	head = ['Query_Index'] + ['Rank_%03d'%i for i in range(1,301)]
	writer.writerow(head)
	for query_id, ans in enumerate(final_ans, 1):
		writer.writerow(['q_%02d'%query_id]+ans)
