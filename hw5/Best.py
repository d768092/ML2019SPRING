import numpy as np
import os,sys
import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169
from skimage.io import imsave

def num_3_digits(n):
    a=n//100
    n-=a*100
    b=n//10
    n-=b*10
    c=n
    return str(a)+str(b)+str(c)

pre_mean = [0.485, 0.456, 0.406]
pre_std = [0.229, 0.224, 0.225]
post_mean = np.array([-0.485, -0.456, -0.406])
post_std = np.array([1/0.229, 1/0.224, 1/0.225])
def de_preprocess(image):
    img_np = image.detach().numpy()
    img_np = img_np.reshape(3, 224, 224)
    img_np = np.transpose(img_np, axes = [1, 2, 0])
    img_np /= post_std
    img_np -= post_mean
    img_np = np.round(np.clip(img_np, 0, 1)*255).astype('uint8')
    return img_np

input_img_dir=sys.argv[1]
output_img_dir=sys.argv[2]
eps=1
wrong=0
num_imgs=200

model = resnet50(pretrained=True)
model.eval()
criterion = nn.CrossEntropyLoss()

for i in range(num_imgs):
    img_file=os.path.join(input_img_dir,num_3_digits(i)+'.png')
    image = Image.open(img_file)
    preprocess = transform.Compose([transform.ToTensor(), transform.Normalize(mean = pre_mean, std = pre_std)])
    image = preprocess(image)
    image = image.unsqueeze(0)
    image.requires_grad = True
    
    # set gradients to zero
    zero_gradients(image)
    
    output = model(image)
    target_label = torch.zeros((1, ))
    target_label[0] = output.argmax()
    before = output.argmax()
    loss = criterion(output, target_label.long())
    loss.backward()
    grad = image.grad.sign_()/255/0.23
    wrong=False
    for j in range(100):
        # add epsilon to image
        image = image + eps * grad
        after = model(image).argmax()
        if before!=after:
            print(i,j)
            wrong=True
            break
    if not wrong:
        print(i,"failed")
    img_output = de_preprocess(image)
    output_file=os.path.join(output_img_dir,num_3_digits(i)+'.png')
    imsave(output_file, img_output)
        
    