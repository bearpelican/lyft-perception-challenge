
# coding: utf-8

# In[35]:


import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
torch.backends.cudnn.benchmark=True


# ## Models
class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()

class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = nn.Conv2d(x_in,  x_out,  1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        
    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p,x_p], dim=1)
        return self.bn(F.relu(cat_p, inplace=True))

class Unet34(nn.Module):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2,4,5,6]]
        self.up1 = UnetBlock(512,256,256)
        self.up2 = UnetBlock(256,128,256)
        self.up3 = UnetBlock(256,64,256)
        self.up4 = UnetBlock(256,64,256)
        self.up5 = UnetBlock(256,3,16)
        self.up6 = nn.ConvTranspose2d(16, 3, 1)
        
    def forward(self,x):
        inp = x
        x = F.relu(self.rn(x), inplace=True)
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x, inp)
        x = self.up6(x)
        return x
    
    def close(self):
        for sf in self.sfs: sf.remove()

class UnetModel():
    def __init__(self,model,name='unet'):
        self.model,self.name = model,name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [lr_cut]))
        return lgs + [children(self.model)[1:]]

def cut_model(m, cut):
    return list(m.children())[:cut] if cut else [m]

def get_base():
    layers = cut_model(f(True), cut)
    return nn.Sequential(*layers)

model_meta = {
    resnet18:[8,6], resnet34:[8,6], resnet50:[8,6], resnet101:[8,6], resnet152:[8,6]
}

# Load model
f = resnet34
cut,lr_cut = model_meta[f]
m_base = get_base()
m = Unet34(m_base)
PATH = Path('../data/Train')

cuda_enabled = torch.cuda.is_available()
model_path = str(Path.cwd()/'600urn-multi.h5')
if cuda_enabled:
    m = m.cuda()
    m.load_state_dict(torch.load(model_path))
else:
    m.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))


# Process video
file = sys.argv[-1]
# file = 'test_video.mp4'

if file == 'demo.py':
  print ("Error loading video")
  quit

# Define encoder function
def encode(array):
	pil_img = Image.fromarray(array)
	buff = BytesIO()
	pil_img.save(buff, format="PNG")
	return base64.b64encode(buff.getvalue()).decode("utf-8")

video = skvideo.io.vread(file)

imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
def normalize(x):
    x = x.astype(np.float32)
    if np.mean(x) > 1:
        x /= 255
    m,s = imagenet_stats
    x -= m
    x /= s
    return x

def preprocess(video):
    f1 = normalize(video)
    f1 = np.rollaxis(f1, 3, 1)
    return f1

video = preprocess(video)
results = []
answer_key = {}
bs = 4
for i in range(0,video.shape[0],bs):
    f1 = video[i:i+bs]
    f1 = np.pad(f1, [(0,0),(0,0),(0,8),(0,0)], mode='constant')
    
    xv = torch.autograd.Variable(torch.from_numpy(f1).contiguous().float())
    if cuda_enabled:
        xv = xv.cuda()
    preds = m(xv)
    mx,idx = torch.max(preds, 1)
    idx = idx[:,:-8,:]
    
    # Frame numbering starts at 1
    frame_idx = 1+i
    for frame in idx:
        # Look for red cars :)
        frame = frame.data.cpu().numpy()
        binary_car_result = (frame==1).astype('uint8')

        # Look for road :)
        binary_road_result = (frame==2).astype('uint8')

        answer_key[frame_idx] = [encode(binary_car_result), encode(binary_road_result)]

        # Increment frame
        frame_idx+=1

# Print output in proper json format
print (json.dumps(answer_key))

