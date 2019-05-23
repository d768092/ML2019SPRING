import numpy as np
import os
import sys
from skimage import io

def process(img):
    M = img.copy()
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

input_path=sys.argv[1]
filelist = os.listdir(input_path) 

# Record the shape of images
img_shape = io.imread(os.path.join(input_path,filelist[0])).shape 
img_data = []
for filename in filelist:
    img = io.imread(os.path.join(input_path,filename))
    img_data.append(img.flatten())

img_data = np.array(img_data).astype('float32')
mean = np.mean(img_data, axis=0)
img_data -= mean
u, s, v = np.linalg.svd(img_data, full_matrices=False)

img = sys.argv[2]
# Load image & Normalize
picked_img = io.imread(os.path.join(input_path,img))  
picked_img = picked_img.flatten().astype('float32') 
picked_img -= mean

V = v[0:5]

# Reconstruction
reconstruct = process((picked_img.dot(V.T)).dot(V) + mean)
io.imsave(sys.argv[3], reconstruct.reshape(img_shape))