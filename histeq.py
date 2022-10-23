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
rgb_white_block=cv2.UMat(rgb_white_block)


def equalize_hist_color(img):
    channels = cv2.split(img)
    eq_channels = []
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch))
    eq_image = cv2.merge(eq_channels)
    return eq_image


if __name__ == '__main__':


    path = "result-dark/our/rgb"

    paths = glob.glob(os.path.join(path, '*' + '.jpg'))
    paths += glob.glob(os.path.join(path, '*' + '.png'))
    img_all=rgb_white_block
    i=0
    time0=time.time()
    for path in paths:
        i+=1
        image=cv2.resize(cv2.imread(path),(512,512))
        cv2.imwrite("pinjie.jpg",equalize_hist_color(image))
        print((time.time()-time0)/i)


