import numpy as np
import pandas as pd
import sys
from keras import backend as K
from keras.models import load_model
from lime import lime_image
from skimage.segmentation import slic
import matplotlib.pyplot as plt

def deprocessImage(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    # print(x.shape)
    return x

def makeNormalize(x):
    
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)


def trainGradAscent(intIterationSteps, arrayInputImageData, targetFunction):
    """
    Implement gradient ascent in targetFunction
    """
    listFilterImages = []
    floatLearningRate = 1e-2
    for i in range(intIterationSteps):
        floatLossValue, arrayGradientsValue = targetFunction([arrayInputImageData, 0])
        arrayInputImageData += arrayGradientsValue * floatLearningRate
    return arrayInputImageData

def predict(image):
    image=np.mean(image.reshape(-1,48,48,3),axis=3).reshape(-1,48,48,1)
    return model.predict(image).reshape(-1,7)

def segmentation(image):
    return slic(image)

intNumberSteps=160
intIterationSteps=200
intChooseId=76
intFilters = 64

plt.switch_backend('agg')
data=pd.read_csv(sys.argv[1])
x_data=data['feature'].values
x_train=np.array([[float(i)/255 for i in feature.split()] for feature in x_data]).reshape(-1,48,48,1)
x_train_rgb=np.concatenate((x_train,x_train,x_train),axis=-1)
y_data=data['label']
model=load_model('model.h5')
x_labels=model.predict_classes(np.mean(x_train_rgb,axis=3).reshape(-1,48,48,1))

idx=[-1 for i in range(7)]
for i in range(len(y_data)):
    if y_data[i]==x_labels[i]:
        if idx[y_data[i]]==-1:
            idx[y_data[i]]=i

inputImage = model.input
collectlayer = model.layers[0].output
listFilterImages = []
fig = plt.figure(figsize=(16, 17))

for i in range(intFilters):
    #print('filter:',i)
    np.random.seed(76)
    arrayInputImage = np.random.random((1, 48, 48, 1)) # random noise
    tensorTarget = K.mean(collectlayer[:, :, :, i])

    tensorGradients = makeNormalize(K.gradients(tensorTarget, inputImage)[0])
    targetFunction = K.function([inputImage, K.learning_phase()], [tensorTarget, tensorGradients])

    # activate filters
    listFilterImages.append(trainGradAscent(intIterationSteps, arrayInputImage, targetFunction))
    ax = fig.add_subplot(intFilters/8, 8, i+1)
    arrayRawImage = listFilterImages[i].squeeze()
    ax.imshow(deprocessImage(arrayRawImage), cmap="Blues")
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.xlabel("filter {}".format(i))
    plt.tight_layout()
plt.savefig(sys.argv[2]+'fig2_1.jpg')
print('get fig2_1.jpg')

collecttarget=K.function([inputImage,K.learning_phase()],[collectlayer])
arrayPhoto = x_train[intChooseId].reshape(1, 48, 48, 1)
listLayerImage = collecttarget([arrayPhoto, 0]) # get the output of that layer list (1, 1, 48, 48, 64)

fig = plt.figure(figsize=(16, 17))
for i in range(intFilters):
    #print('filter:',i)
    ax = fig.add_subplot(intFilters/8, 8, i+1)
    ax.imshow(listLayerImage[0][0, :, :, i], cmap="Blues")
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.xlabel("filter {}".format(i))
    plt.tight_layout()
plt.savefig(sys.argv[2]+'fig2_2.jpg')
print('get fig2_2.jpg')

explainer=lime_image.LimeImageExplainer()
for i in range(len(idx)):
    if idx[i]==-1:
        print('no class '+str(i))
        continue
    np.random.seed(76)
    pred=model.predict_classes(x_train[idx[i]].reshape(-1,48,48,1))[0]
    tensortarget=model.output[:,pred]
    tensorgrad=K.gradients(tensortarget,inputImage)[0]
    fn=K.function([inputImage, K.learning_phase()],[tensorgrad])
    grad=fn([x_train[idx[i]].reshape(-1,48,48,1)])[0].reshape(48,48,-1)
    grad=np.max(np.abs(grad),axis=-1,keepdims=True)
    grad=(grad-np.mean(grad))/(np.std(grad)+1e-7)
    grad=grad*0.1+0.5
    grad=np.clip(grad,0,1)
    heatmap=grad.reshape(48,48)
    fig=plt.figure(figsize=(10,3))
    
    ax=fig.add_subplot(1,3,1)
    axx=ax.imshow((x_train[idx[i]]*255).reshape(48,48), cmap='gray')
    plt.tight_layout()

    ax=fig.add_subplot(1,3,2)
    axx=ax.imshow(heatmap,cmap=plt.cm.jet)
    plt.colorbar(axx)
    plt.tight_layout
    
    threshold=0.5
    maskedmap=(x_train[idx[i]]*255).reshape(48,48)
    maskedmap[np.where(heatmap<threshold)]=np.mean(maskedmap)
    ax=fig.add_subplot(1,3,3)
    axx=ax.imshow(maskedmap,cmap='gray')
    plt.tight_layout()
    plt.savefig(sys.argv[2]+'fig1_'+str(i)+'.jpg')
    print('get fig1_'+str(i)+'.jpg')
    
    explanation=explainer.explain_instance(image=x_train_rgb[idx[i]], classifier_fn=predict, segmentation_fn=segmentation)
    image, mask=explanation.get_image_and_mask(label=x_labels[idx[i]],
    positive_only=False, hide_rest=False, num_features=5, min_weight=0.0)
    plt.imsave(sys.argv[2]+'fig3_'+str(i)+'.jpg',image)
    print('get fig3_'+str(i)+'.jpg')
