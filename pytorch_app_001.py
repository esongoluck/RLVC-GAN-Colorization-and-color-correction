#from visdom import Visdom
#import inputs
import util
import config
import datas
from skimage import color
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2
import eccv16
import train
import show
import my_losses
from my_dataset import My_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from siggraph17 import SIGGRAPHGenerator
import siggraph17
from torchvision import transforms
#torch.cuda.current_device()
#torch.cuda._initialized = True


device=torch.device(config.Config['device'])

if __name__ == '__main__':
    print('use gpu? ',torch.cuda.is_available(),'     gpu nums: ',torch.cuda.device_count())
    conf=config.Config
    db=My_dataset(config.Config,conf["batch_size"],'train')
    dat=datas.datas()
    image_size=conf["image_size"]
    #x,y=next(iter(db))只取一个这样取
    #print(x.shape,y.shape)
    #loader=DataLoader(db,batch_size=2,shuffle=True)#要删掉
    #for x,y in loader:
    #    print(x.shape,y.shape)#要删掉


    #fo = open("imgs/foo.txt", "w")
    #fo.write( "www.runoob.com!\nVery good site!\n")
    
    ## 关闭打开的文件
    #fo.close()

    train.to_train(dat,config.Config)
    dat.model=SIGGRAPHGenerator().to(device)#使用gpu运算，模型放进cuda
    dat.model.load_state_dict(torch.load('best.mdl'))




    impath="C:/Users/luckerr/source/repos/pytorch_app_001/pytorch_app_001/my_dataset/img-ir-s"
    name0=1816
    tmp_lab=torch.zeros(1,4,image_size,image_size, dtype=torch.float32).to(device)
    imgs_out=torch.zeros(32,3,image_size,image_size, dtype=torch.float32).to(device)
    impathi=0
    for i in range(32):
        impathi=os.path.join(impath,str(name0+i).zfill(6)+".jpg")
        imgi = util.load_img(impathi)
        imgi = np.array(imgi,dtype = np.float32)
        if(imgi.ndim==2):
            imgi = np.tile(imgi[:,:,None],3)
        print("imgi_0",imgi)
        imgi=cv2.resize(imgi,(image_size,image_size))
        print("imgi_1",imgi)
        imgi_lab = color.rgb2lab(imgi/255)
        imgi_lab=torch.from_numpy(imgi_lab)
        imgi_lab=imgi_lab.permute(2,0,1)#调换维度，transpose只能对换两个维度，permute可对换多个维度
        print("imgi_lab",imgi_lab)
        print("imgi_lab",imgi_lab.shape)
        tmp_lab[0,0,:,:]=imgi_lab[0,:,:]
        print("tmp_lab",tmp_lab.shape)
        imgs_out[i,:,:,:]=dat.model(tmp_lab)[0,:,:,:]
        tmp_lab[0,1,:,:]=imgs_out[i,0,:,:]
        tmp_lab[0,2,:,:]=imgs_out[i,1,:,:]
        tmp_lab[0,3,:,:]=imgs_out[i,2,:,:]
        #print("!!!!",imgs_out.shape)

        #img_llab=torch.stack([img_L,0*lable_L,0*lable_a,0*lable_b],dim=0)
    util.show_lab_images_32(imgs_out)
    while 1:
        1

    (tens_l_orig, tens_l_rs) = util.preprocess_img(img, HW=(conf["image_size"],conf["image_size"]))#保留原始图像尺寸等信息
    tens_l_rs = tens_l_rs.to(device)

    img_bw = util.postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=0))#原始图像

    tens_l_rs=torch.stack([tens_l_rs,0*tens_l_rs,0*tens_l_rs,0*tens_l_rs],dim=1)#1通道图像变4通道图像
    print("tens_l_rs:",tens_l_rs.cpu())
    tens_out=dat.model(tens_l_rs)
    tens_out=tens_out[0,:,:,:]
    tens_out[0,:,:]=tens_out[0,:,:]
    print("!!!! tens_ab  ",tens_out.cpu().shape, tens_out.cpu())
    #out_img_eccv17 = util.postprocess_tens( dat.model(tens_l_rs)[0,0,:,:].cpu(),  dat.model(tens_l_rs)[0,,:,:].cpu().cpu())#4个维度变3个维度,合并
    #temp000=tens_out.cpu().detach().numpy().transpose((1,2,0))
    print("temp000",temp000.shape)
    #out_img_eccv17=color.lab2rgb(tens_out.cpu().detach().numpy().transpose((1,2,0)))
    #out_img_eccv17=cv2.resize(out_img_eccv17,(tens_l_orig.shape[2],tens_l_orig.shape[1]))
    out_img_eccv17=0
    print("out_img.shape",out_img_eccv17.shape)
    #model2=siggraph17.SIGGRAPHGenerator().to(device)
    #model2=siggraph17.siggraph17_dl().to(device)
    #out_img_eccv16 = util.postprocess_tens(tens_l_orig, model2(tens_l_rs).cpu())
    out_img_eccv16=out_img_eccv17

    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(img_bw)
    plt.title('Input')
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(out_img_eccv16)
    plt.title('Output (my 17)')
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(out_img_eccv17)
    plt.title('Output (down load 17)')
    plt.axis('off')
    plt.show()




