# from visdom import Visdom
# import inputs
import cv2.cuda
from torchstat import stat
import datetime
import traceback
import util
import config
import datas
from skimage import color
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import time
import cv2
import PIL
from PIL import Image
import eccv16
import train
import show
import my_losses
from my_dataset import My_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from models import SIGGRAPHGenerator,myGenerator
import math
import siggraph17
from torchvision import transforms
import torchvision

# torch.cuda.current_device()
# torch.cuda._initialized = True



device = torch.device(config.Config['device'])
bias = np.array([[255.0, 255.0, 255.0]])

rgb_white_block = np.zeros((512, 2, 3), dtype=np.float32) + 1
if __name__ == '__main__':


    path = "result-dark/StableLLVE"

    paths = glob.glob(os.path.join(path, '*' + '.jpg'))
    paths += glob.glob(os.path.join(path, '*' + '.png'))
    img_all=rgb_white_block
    i=0
    for path in paths:
        i+=1
        if i<=3 :
            continue
        image=cv2.resize(cv2.imread(path).astype(np.float32)/255,(512,512))
        print(img_all.shape,image.shape)
        img_all=np.hstack((img_all,image))
        img_all=np.hstack((img_all,rgb_white_block))

    cv2.imwrite("pinjie.jpg",cv2.multiply(img_all,bias))