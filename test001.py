import argparse
import os
import shutil
import time
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import ImageFile





if __name__ == '__main__':

    img=np.zeros((1000,1000,3),dtype=np.float32)
    for i in range(0,1000):
        for j in range(0,1000):
            img[i][j][0]=0.5
            img[i][j][1]=i/1000
            img[i][j][2]=j/1000



    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)  ##########图像转化为RGB
    bias=np.array([[255.0, 255.0, 255.0]])
    img=cv2.multiply(img, bias)
    img=np.clip(img,0,255)
    img=img.astype(np.uint8)#转为uint8
    image = Image.fromarray(img)  # 把图片从cv2格式转换成Image

    plt.figure("Image")  # 图像窗口名称
    plt.imshow(image)
    plt.axis('on')  # 关掉坐标轴为 off
    #plt.title('image')  # 图像题目

    plt.xticks(np.arange(0, 1000,100), np.arange(0, 10,1)/10.0)
    plt.yticks(np.arange(0, 1000,100), np.arange(0, 10,1)/10.0)
    #plt.xticks(np.arange(0, 1000,1))
    #plt.yticks(np.arange(0, 1000,1))
    plt.xlabel('Cb')
    plt.ylabel('Cr')

    # 必须有这个，要不然无法显示
    plt.show()
    print("!!!!")
    print("#####")
    #main()
