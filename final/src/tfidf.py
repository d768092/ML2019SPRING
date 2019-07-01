import numpy as np
import json
import sys

data = np.load("../data", allow_pickle = True)

word_dic = dict()
idf_dic = dict()
now = 1
print("set dictictionary")
for i in data:
	print(now, end = "\r")
	is_in_paragraph = dict()
	for j in i:
		if j not in idf_dic:
			idf_dic[j] = {"idf": 0, "docs": []}
		
		if j not in is_in_paragraph:
			is_in_paragraph[j] = 1
		else:
			is_in_paragraph[j] += 1

	for j in is_in_paragraph:
		idf_dic[j]["idf"] += 1
		idf_dic[j]["docs"].append({"news_%06d"%(now): is_in_paragraph[j]})
	
	now += 1

print("compute idf")
now = 1
for i in idf_dic:
	print(now, end = "\r")
	idf_dic[i]["idf"] = np.log(100000 / (1 + idf_dic[i]["idf"]))
	now += 1

with open("../my_inverted_file", "w") as output:
	json.dump(idf_dic, output)
