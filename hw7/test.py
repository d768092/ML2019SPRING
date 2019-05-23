import numpy as np
import pandas as pd
import os
import sys
import csv
from keras.models import Model
from keras.models import load_model
from skimage import io
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE

testcase1 = pd.read_csv(sys.argv[2])['image1_name'].values
testcase2 = pd.read_csv(sys.argv[2])['image2_name'].values

input_path=sys.argv[1]
train = []
for i in range(1,40001):
    img = io.imread(os.path.join(input_path,str(i).zfill(6)+'.jpg'))/255
    train.append(img)

train = np.array(train)
encoder = load_model('encoder.model')
codes = encoder.predict(train)

cluster = TSNE(n_jobs=2).fit_transform(codes)
#cluster = PCA(n_components=2).fit_transform(codes)
kmeans = KMeans(n_clusters=2).fit(cluster)
with open(sys.argv[3],'w+',newline='') as output:
    writer=csv.writer(output)
    writer.writerow(['id','label'])
    for i in range(len(testcase1)):
        if kmeans.labels_[int(testcase1[i])-1]==kmeans.labels_[int(testcase2[i])-1]:
            writer.writerow([i,1])
        else:
            writer.writerow([i,0])

