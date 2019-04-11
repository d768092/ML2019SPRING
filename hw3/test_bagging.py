import pandas as pd
import numpy as np
import sys
import csv
from keras.models import load_model
from collections import Counter
data=pd.read_csv(sys.argv[1])
test=data['feature'].values
test=np.array([[float(i)/255 for i in feature.split()] for feature in test]).reshape(-1,48,48,1)

bags=int(sys.argv[3])
predictions=[]
for i in range(bags):
    model=load_model(sys.argv[2]+'_'+str(i))
    predictions.append(model.predict_classes(test))

predictions=np.array(predictions)
predictions=[ Counter(predictions[:,i]).most_common(1)[0][0] for i in range(test.shape[0]) ]
with open(sys.argv[4],'w+',newline='') as output:
    writer=csv.writer(output)
    writer.writerow(['id','label'])
    for i in range(len(predictions)):
        writer.writerow([i,predictions[i]])
print(predictions)
