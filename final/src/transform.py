import json
import jieba
from argparse import ArgumentParser
import sys
import numpy as np

parser = ArgumentParser()
parser.add_argument("-i", "--url2content", default='../url2content.json', dest = "url2content", help = "Pass in a .json file.")

jieba.set_dictionary("dict.txt.big")

args = parser.parse_args()
# load inverted file
with open(args.url2content) as f:
	invert_file = json.load(f)

data = []
cnt = 0
for i in invert_file.values():
	data.append(jieba.lcut(i.replace("\n", ""), cut_all = False))
	print(cnt, end = "\r")
	cnt += 1

print(np.array(data))
np.save("../data", np.array(data))
