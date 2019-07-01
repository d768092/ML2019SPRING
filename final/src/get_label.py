import numpy as np
import sys
import json

def sortSecond(val):
	return val[1]

data = np.delete(np.genfromtxt("../TD.csv", delimiter = ",", dtype = np.str), 0, 0)

category = dict()

for i in data:
	if i[0] == "堅決反對政府舉債發展前瞻建設計畫":
		title = '同意政府舉債發展前瞻建設計畫'
		if not int(i[2]) == 0:
			continue
		if title in category:
			category[title].append([i[1], i[2]])
		else:
			category[title] = [[i[1], i[2]]]

	else:
		if int(i[2]) == 0:
			continue
		if i[0] in category:
			category[i[0]].append([i[1], i[2]])
		else:
			category[i[0]] = [[i[1], i[2]]]

for i in category:
	category[i].sort(reverse = True, key = sortSecond)

with open("../TD.json", "w") as output:
	json.dump(category, output)
