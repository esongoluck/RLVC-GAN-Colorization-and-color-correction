import os
import cv2
import config
import torch
import util
import numpy
from numpy import *
import torchvision
import ssl
from skimage import color

from torch.utils.data import DataLoader
ssl._create_default_https_context = ssl._create_unverified_context

train_data = torchvision.datasets.CIFAR10(root="/data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
train_dataloader = DataLoader(train_data, batch_size=64)
#print("!!!!!!!!!!!!!!cifer10   ",train_dataloader)
def get_img_list(dir, firelist, ext=None):
    newdir = dir
    if os.path.isfile(dir):  # 如果是文件
        if ext is None:
            firelist.append(dir)
        elif ext in dir[-3:]:
            firelist.append(dir)
    elif os.path.isdir(dir):  # 如果是目录
        for s in os.listdir(dir):
            newdir = os.path.join(dir, s)
            get_img_list(newdir, firelist, ext)

    return firelist

def read_img():
    image_path = config.config['image_list']
    imglist = get_img_list(image_path, [], 'png')
    imgall = []
    for imgpath in imglist:
        img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        img=cv2.resize(img,(256,256))
        print(img,"!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
        img = img[:,:,::-1]
        print(img,"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        #img=torch.tensor(img)
        imgall.append(img)
    
    return imgall

def img_to_datset(dat,imgall):
    for img in imgall:
        (tens_l_orig, tens_l_rs) = util.preprocess_img(img, HW=(256,256))
        print(tens_l_orig.shape)


def input_dataset(dat):
    imgall=read_img()
    img_to_datset(dat,imgall)
