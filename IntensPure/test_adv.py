# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ft_net
from utils import fuse_all_conv_bn
#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../1.datasets/Market-1501/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--revised_query', type=str)

opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader) # for the new pyyaml via 'conda install pyyaml'
opt.stride = config['stride']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 751 

if 'linear_num' in config:
    opt.linear_num = config['linear_num']

str_ids = opt.gpu_ids.split(',')
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
h, w = 256, 128

data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = test_dir

revised_query = opt.revised_query
revised_query_f = 'query_f_'+revised_query
if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=0) for x in ['gallery']}
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery',revised_query]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=0) for x in ['gallery',revised_query]}
class_names = image_datasets[revised_query].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    #features = torch.FloatTensor()
    count = 0
    if opt.linear_num <= 0:
        opt.linear_num = 2048

    for iter, data in enumerate(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img) 
                ff += outputs
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
           
        if iter == 0:
            features = torch.FloatTensor( len(dataloaders.dataset), ff.shape[1])
                       
        start = iter*opt.batchsize
        end = min( (iter+1)*opt.batchsize, len(dataloaders.dataset))
        features[ start:end, :] = ff
        
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets[revised_query].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
model_structure = ft_net(opt.nclasses, stride = opt.stride, ibn = opt.ibn, linear_num=opt.linear_num)
model = load_network(model_structure)
model.classifier.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

model = fuse_all_conv_bn(model)

# Extract feature
since = time.time()
with torch.no_grad():
    gallery_feature = extract_feature(model,dataloaders['gallery'])
    query_feature = extract_feature(model,dataloaders[revised_query])
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.2f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam, revised_query_f:query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('mat/'+revised_query+'.mat',result)
print("save mat file: ",revised_query+'.mat')
result = './model/%s/'%opt.name+revised_query+'.txt'
os.system('python evaluate_adv_gpu.py --revised_query {}'.format(revised_query))