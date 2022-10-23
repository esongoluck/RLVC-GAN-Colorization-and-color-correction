import cv2.cuda
import torch
import os,glob
import random,csv
import util
import config
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from skimage import color
import cv2
import time
import datas
import matplotlib.pyplot as plt
class My_dataset(Dataset):
    def __init__(self,conf,resize,mode):
        super(My_dataset,self).__init__()
        self.image_size=conf["image_size"]
        self.root=conf["date_path"]
        self.train_x_path=self.root+"/train"+conf["x_path"]
        self.train_y_path=self.root+"/train"+conf["y_path"]
        self.val_x_path=self.root+"/val"+conf["x_path"]
        self.val_y_path=self.root+"/val"+conf["y_path"]
        self.test_x_path=self.root+"/test"+conf["x_path"]
        self.test_y_path=self.root+"/test"+conf["y_path"]
        self.resize=resize
        self.name2label={}
        self.conf=conf
        self.mode=mode
        #for name in sorted(os.listdir(os.path.join(self.root))):
        #    if not os.path.isdir(os.path.join(self.root,name)):#查看路径是否为一个目录
        #        continue
        #    self.name2label[name]=len(self.name2label.keys())
        self.train_x,self.train_y=self.load_csv(os.path.join(self.root+"/train",'image.csv'),"train_csv",self.train_x_path,self.train_y_path)
        self.val_x,self.val_y=self.load_csv(os.path.join(self.root+"/val",'image.csv'),"val_csv",self.val_x_path,self.val_y_path)
        self.test_x,self.test_y=self.load_csv(os.path.join(self.root+"/test",'image.csv'),"test_csv",self.test_x_path,self.test_y_path)
        label_train=0.8
        label_test=0.9
        self.non_x=torch.tensor([[[0.5]*resize]*resize]*3)
        self.non_y=torch.tensor([[[0.5]*resize]*resize]*9)
        if mode=='train':#80%
            self.xs=self.train_x
            self.ys=self.train_y
        elif mode=='val':#1%
            self.xs=self.val_x
            self.ys=self.val_y
        elif mode=='test':#1%
            self.xs=self.test_x
            self.ys=self.test_y

        # if mode=='train':#80%
        #     self.images=self.images[:int(label_train*len(self.images))]
        #     self.labels=self.labels[:int(label_train*len(self.labels))]
        # elif mode=='val':#80-90%
        #     self.images=self.images[int(label_train*len(self.images)):int(label_test*len(self.images))]
        #     self.labels=self.labels[int(label_train*len(self.images)):int(label_test*len(self.labels))]
        # else:#90%
        #     self.images=self.images[int(label_test*len(self.images)):]
        #     self.labels=self.labels[int(label_test*len(self.images)):]


    def load_csv(self,filename,csv_type,x_path,y_path):##添加一堆对比
        if self.conf[csv_type]==1 or not os.path.exists(filename):
            self.conf[csv_type]=0#csv文件不重复初始化
            xs=[]
            images2=[]#筛选后存这里(同时具备标签和红外图像的图像)
            xs+=glob.glob(os.path.join(x_path,'*.png'))
            xs+=glob.glob(os.path.join(x_path,'*.jpg'))
            xs+=glob.glob(os.path.join(x_path,'*.jpeg'))
            #pathi,namei=os.path.split(images[0])#拆分路径、路径名
            ######################################################random.shuffle(xs)#################在这里《需要》对路径顺序进行随机排序
            with open(filename,mode='w',newline='')as io_f:
                writer=csv.writer(io_f)
                for x in xs:
                    name=x.split(os.sep)[-1]
                    name,_,type=name.rpartition('.')
                    name_1=str(int(name)-1).zfill(6)
                    name_2=str(int(name)-2).zfill(6)

                    x=x_path+"/"+name+"."+type#查看当前图是否存在
                    y=y_path+"/"+name+"."+type

                    x_1=x_path+"/"+name_1+"."+type#查看前一张图是否存在
                    y_1=y_path+"/"+name_1+"."+type

                    x_2=x_path+"/"+name_2+"."+type#查看前2张图是否存在
                    y_2=y_path+"/"+name_2+"."+type
                    if not os.path.exists(x) or not os.path.exists(x_1) or not os.path.exists(x_2):
                        continue
                    if not os.path.exists(y) or not os.path.exists(y_1) or not os.path.exists(y_2):
                        continue
                    writer.writerow([x,y])

        xs,ys=[],[]
        with open(filename) as io_f:    #xs，ys 存重整后的x，y文件路径

            reader=csv.reader(io_f)
            for row in reader:
                x,y=row
                xs.append(x)
                ys.append(y)

        assert len(xs)==len(ys)
        return xs,ys


    def __len__(self):
        return len(self.xs)
    def read_y(self,y):
        name, _, type = y.rpartition('.')
        path, _, name = name.rpartition('/')
        name_1=str(int(name) - 1).zfill(6)
        name_2=str(int(name) - 2).zfill(6)
        y_1=os.path.join(path,name_1+'.'+type)
        y_2=os.path.join(path,name_2+'.'+type)
        y=cv2.imread(y)
        y_1=cv2.imread(y_1)
        y_2=cv2.imread(y_2)
        if y is None:#假如找到一个空的图，直接返回空图
            print("empty y  !",name,os.path.join(path,name+'.'+type))
            return self.non_y
        if y_1 is None:
            print("empty y_1  !",name_1)
            return self.non_y
        if y_2 is None:
            print("empty y_2  !",name_2)
            return self.non_y

        y = y.astype(np.float32) / 255  #转化为float
        y_1 = y_1.astype(np.float32) / 255  #转化为float
        y_2 = y_2.astype(np.float32) / 255  #转化为float

        y=cv2.UMat(y)                   #存进gpu
        y_1=cv2.UMat(y_1)                   #存进gpu
        y_2=cv2.UMat(y_2)                   #存进gpu

        y=cv2.warpPerspective(y,self.conf["tr_mat_y"],(self.resize,self.resize))
        y_1=cv2.warpPerspective(y_1,self.conf["tr_mat_y"],(self.resize,self.resize))
        y_2=cv2.warpPerspective(y_2,self.conf["tr_mat_y"],(self.resize,self.resize))
        #cv2.imwrite("imgout/y.jpg",y.get()*255)
        y=cv2.cvtColor(y, cv2.COLOR_BGR2YCrCb);#转化为lab
        y_1=cv2.cvtColor(y_1, cv2.COLOR_BGR2YCrCb);#转化为lab
        y_2=cv2.cvtColor(y_2, cv2.COLOR_BGR2YCrCb);#转化为lab
        y=cv2.merge([y_2,y_1,y]).get()

        return torch.tensor(y).permute(2,0,1)#转换为tensor并调换维度
    def read_x(self,x):

        name, _, type = x.rpartition('.')
        path, _, name = name.rpartition('/')
        name_1=str(int(name) - 1).zfill(6)
        name_2=str(int(name) - 2).zfill(6)

        if self.mode=="train":
            1#print(int(name),end=" ")
        x_1=os.path.join(path,name_1+'.'+type)
        x_2=os.path.join(path,name_2+'.'+type)
        x=cv2.imread(x)
        x_1=cv2.imread(x_1)
        x_2=cv2.imread(x_2)
        if x is None:#假如找到一个空的图，直接返回空图
            print("empty x !",name)
            return self.non_x
        if x_1 is None:
            print("empty x_1  !",name_1)
            return self.non_x
        if x_2 is None:
            print("empty x_2  !",name_2)
            return self.non_x

        x = x.astype(np.float32) / 255  #转化为float
        x_1 = x_1.astype(np.float32) / 255  #转化为float
        x_2 = x_2.astype(np.float32) / 255  #转化为float

        x=cv2.UMat(x)                   #存进gpu
        x_1=cv2.UMat(x_1)                   #存进gpu
        x_2=cv2.UMat(x_2)                   #存进gpu

        x=cv2.cvtColor(x, cv2.COLOR_BGR2GRAY);#转化为灰度
        x_1=cv2.cvtColor(x_1, cv2.COLOR_BGR2GRAY);#转化为灰度
        x_2=cv2.cvtColor(x_2, cv2.COLOR_BGR2GRAY);#转化为灰度
        x=cv2.warpPerspective(x,self.conf["tr_mat_x"],(self.resize,self.resize))
        x_1=cv2.warpPerspective(x_1,self.conf["tr_mat_x"],(self.resize,self.resize))
        x_2=cv2.warpPerspective(x_2,self.conf["tr_mat_x"],(self.resize,self.resize))
        #cv2.imwrite("imgout/x.jpg",x.get()*255)
        x=cv2.merge([x_2,x_1,x]).get()

        return torch.tensor(x).permute(2,0,1)
    def __getitem__(self,idx):#读取第i张图
        if self.mode=="train":
            1#idx=min(idx+5100,19000)
        x,y=self.xs[idx],self.ys[idx]#当前图像路径
        tf_x=transforms.Compose([
            transforms.ColorJitter(brightness=0.8,contrast=0.5,saturation=0.5)

        ])

        y=self.read_y(y)
        x=self.read_x(x)


        if self.mode=="train":
            #util.imwrite_tensor_l("imgout/x0.jpg", x[2:3, :, :])
            x=tf_x(x)
            #util.imwrite_tensor_l("imgout/x1.jpg", x[2:3, :, :])
            #print("changed!!!!")
        return x,y
