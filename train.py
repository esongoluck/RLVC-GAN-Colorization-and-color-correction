import torch
from torch import optim,nn
import time
import torchvision
from torch.utils.data import DataLoader
import datas
from my_dataset import My_dataset
#import my_optimizers
from eccv16 import ECCVGenerator
import eccv16
#from siggraph17 import SIGGRAPHGenerator
from models import SIGGRAPHGenerator_1,SIGGRAPHGenerator,NLayerDiscriminator,NLayerGenerator,myGenerator
import siggraph17
import config
import models
import my_losses
import cv2
#from visdom import Visdom
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import util
#print("train_ac: " , torch.cuda.is_available(),torch.cuda.device_count())
adversarial_loss = torch.nn.BCELoss()
#my_loss=nn.CrossEntropyLoss()
#direct_Loss=my_losses.direct_Loss()
device=torch.device(config.Config['device'])
#my_loss = nn.BCELoss()
#my_loss=nn.MSELoss().to(device)
from torchstat import stat
import torchvision.models as models
import apex                     #不能直接pip，pip到的不对，需要使用英伟达github上的apex

from apex import amp



def evalute(G_model,loader):#计算统计数据，无实际作用
    mse_Loss=my_losses.mse_Loss()
    psnr_Loss=my_losses.psnr_Loss(MaxNum=1.0)
    ssim_Loss=my_losses.ssim_Loss()
    direct_Loss=my_losses.direct_Loss()
    G_Loss=my_losses.G_Loss()
    #cyc_Loss=my_losses.cyc_Loss()
    correct=0

    total=len(loader.dataset)
    mse_loss_mean=0
    psnr_loss_mean=0
    ssim_loss_mean=0
    direct_loss_mean=0
    sumi=0
    for x,y in loader:
        if x.shape[0]<config.Config["batch_size"]:
            continue
        x,y=x.to(device),y.to(device)
        #if config.Config["half"] == "True":
        #    x, y = x.half(), y.half()
        #time0=time.time()
        yp=G_model(x).detach()
        if sumi==1:
            #print(x[0][2:3,:,:].shape)
            util.imwrite_tensor_l("imgout/x.jpg",x[0][2:3,:,:])
            util.imwrite_tensor_lab("imgout/y.jpg",y[0][6:9,:,:])
            util.imwrite_tensor_lab("imgout/yp.jpg",yp[0])
        #print("time",time.time()-time0)

        rgb_false = yp[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        rgb_false = cv2.cvtColor(rgb_false, cv2.COLOR_YCrCb2RGB)  ##########图像转化为BGR
        rgb_false=torch.tensor(rgb_false).permute(2, 0, 1)[None,:,:,:].to(device)
        rgb_true = y[0][6:9,:,:].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        rgb_true = cv2.cvtColor(rgb_true, cv2.COLOR_YCrCb2RGB)  ##########图像转化为BGR
        rgb_true = torch.tensor(rgb_true).permute(2, 0, 1)[None,:,:,:].to(device)
        #print(rgb_false.shape,rgb_false.shape)
        mse_loss_mean+=mse_Loss(rgb_false,rgb_true)
        psnr_loss_mean+=psnr_Loss(rgb_false,rgb_true)
        ssim_loss_mean+=ssim_Loss(rgb_false,rgb_true)#########################

        #mse_loss_mean+=mse_Loss(yp,y[:,6:9,:,:])
        #psnr_loss_mean+=psnr_Loss(yp,y[:,6:9,:,:])
        #ssim_loss_mean+=ssim_Loss(yp,y[:,6:9,:,:])#########################
        direct_loss_mean+=direct_Loss(yp,y)
        sumi+=1
    res=torch.Tensor([mse_loss_mean,psnr_loss_mean,ssim_loss_mean,direct_loss_mean])/total*config.Config["batch_size"]
    return res
    #return correct/total


def to_train(dat,conf):
    batchsz=conf["batch_size"]
    LR=0.00001/conf["batch_size"]
    LR_G=LR*conf["batch_size"]
    LR_D=LR*conf["batch_size"]
    LR_C=LR*conf["batch_size"]
    epochs=conf["epochs"]
    image_size=conf["image_size"]
    torch.manual_seed(1234)


    train_db=My_dataset(config.Config,image_size,mode='train')
    val_db=My_dataset(config.Config,image_size,mode='val')
    test_db=My_dataset(config.Config,image_size,mode='test')

    dat.train_loader=DataLoader(train_db,batch_size=batchsz,num_workers=8)
    dat.val_loader=DataLoader(val_db,batch_size=batchsz,num_workers=8)
    dat.test_loader=DataLoader(test_db,batch_size=batchsz,num_workers=8)

    dat.G_model=myGenerator().to(device)#使用gpu运算，模型放进cuda
    dat.D_model=myGenerator(in_nums=6,out_nums=1,use_sigmoid=conf["MD_sig"]).to(device)
    dat.C_model=myGenerator(in_nums=3,out_nums=1).to(device)

    ########dat.model=siggraph17.siggraph17_dl().to(device)#读训练好的模型

    D_model=dat.D_model
    G_model=dat.G_model
    C_model=dat.C_model

    D_Loss=my_losses.D_Loss().to(device)
    G_Loss=my_losses.G_Loss().to(device)
    C_Loss=my_losses.C_Loss().to(device)


    # G_optimizer=optim.Adam(G_model.parameters(),lr=LR_G,betas=(0.5,0.99),eps=1e-3)
    # D_optimizer=optim.Adam(D_model.parameters(),lr=LR_D,betas=(0.5,0.99),eps=1e-3)
    # C_optimizer=optim.Adam(C_model.parameters(),lr=LR_C,betas=(0.5,0.99),eps=1e-3)
    G_optimizer=optim.RMSprop(G_model.parameters(),lr=LR_G,eps=1e-3)
    D_optimizer=optim.RMSprop(D_model.parameters(),lr=LR_G,eps=1e-3)
    C_optimizer=optim.RMSprop(C_model.parameters(),lr=LR_G,eps=1e-3)
    # stat(G_model.cpu(), (3, 512, 512))
    # stat(D_model.cpu(), (6, 512, 512))
    # stat(C_model.cpu(), (3, 512, 512))
    if conf["device"]=="cuda:0":
        [D_model, G_model, C_model], [D_optimizer, G_optimizer, C_optimizer] = apex.amp.initialize(
            [D_model, G_model, C_model], [D_optimizer, G_optimizer, C_optimizer], opt_level="O3")

    #return
    if conf["half"] == "True":
        1
        # print("!!!!!!!!!!!!!!!!")
        # D_model = D_model.half()
        # G_model = G_model.half()
        # C_model = C_model.half()

        #D_Loss = D_Loss.half()
        #G_Loss = G_Loss.half()
        #C_Loss = C_Loss.half()

        # D_optimizer = my_optimizers.Adam_half(D_model.parameters(), lr=LR_D, betas=(0.9, 0.99), eps=1e-4)
        # G_optimizer = my_optimizers.Adam_half(G_model, lr=LR_G, betas=(0.9, 0.99), eps=1e-4)
        # C_optimizer = my_optimizers.Adam_half(C_model.parameters(), lr=LR_C, betas=(0.9, 0.99), eps=1e-4)

        #[D_model,G_model,C_model], [D_optimizer,G_optimizer,C_optimizer] = apex.amp.initialize([D_model,G_model,C_model], [D_optimizer,G_optimizer,C_optimizer],opt_level="O3")
    #print(dat.G_model.state_dict())
    best_acc,best_epoch=1000,-1
    print('best_acc start :',best_acc,'best epoch:',best_epoch)
    show_sum=0
    for epoch in range(epochs):
        for step,(x,y) in enumerate(dat.train_loader):
            #if True :

            #print("x0 ",x.shape)
            if x.shape[0]<conf["batch_size"]:
                continue

            x,y=x.to(device), y.to(device)

            #if conf["half"]=="True":
            #    x, y = x.half(), y.half()

            #with autocast():
            #print("!!!   0   ",D_model.state_dict())

            #训练分类器
            z_D=D_model(torch.cat((x,y[:,6:,:,:]),dim=1))
            yp=G_model(x).detach()
            z_Dp=D_model(torch.cat((x,yp),dim=1))

            D_optimizer.zero_grad()

            #with autocast():
            D_loss=D_Loss(z_D,z_Dp)#大写L是类，小写l是数值
            if conf["device"]=="cuda:0":
                if conf["half"] == "True":
                    with amp.scale_loss(D_loss, D_optimizer) as scaled_D_loss:
                        scaled_D_loss.backward()
                else:
                    D_loss.backward()
            else:
                D_loss.backward()
            D_optimizer.step()
            # 训练cyc
            x_C = C_model(y[:,6:,:,:])

            C_optimizer.zero_grad()

            #with autocast():
            C_loss = C_Loss(x[:,2:3,:,:], x_C)  # 大写L是类，小写l是数值
            if conf["device"]=="cuda:0":
                if conf["half"] == "True":
                    with amp.scale_loss(C_loss, C_optimizer) as scaled_C_loss:
                        scaled_C_loss.backward()
                else:
                    C_loss.backward()
            else:
                C_loss.backward()
            C_optimizer.step()
            #print(".",end=" ")


            #训练生成器

            yp=G_model(x)
            z_Dp=D_model(torch.cat((x,yp),dim=1))

            x_Cp=C_model(yp)
            #print("!!!   2   ",x_Cp,torch.mean(x_Cp))

            G_optimizer.zero_grad()

            #with autocast():
            G_loss=G_Loss(y,yp,z_Dp,x[:,2:3,:,:],x_Cp)

            if conf["device"]=="cuda:0":
                if conf["half"] == "True":
                    with amp.scale_loss(G_loss, G_optimizer) as scaled_G_loss:
                        scaled_G_loss.backward()
                else:
                    G_loss.backward()
            else:
                G_loss.backward()
            G_optimizer.step()

            #torch.save(dat.G_model.state_dict(), 'best.mdl')
            #print("!!!   3   ",G_model.state_dict())
            #print("!!!   4   ",C_model.state_dict())

            if show_sum%50==0 and show_sum!=0:
                util.imwrite_tensor_l("imgout/tran_x.jpg", x[0][2:3, :, :][0])
                util.imwrite_tensor_lab("imgout/tran_y.jpg", y[0][6:9, :, :])
                util.imwrite_tensor_lab("imgout/tran_yp.jpg", yp[0].detach())
                best_acc, best_epoch = evalute(dat.G_model, dat.val_loader), epoch
                print('best_acc:', best_acc, 'best epoch:', best_epoch,show_sum)
                if not torch.isnan(best_acc[0]):
                    torch.save(dat.G_model.state_dict(), 'best.mdl')
                    torch.save(dat.D_model.state_dict(), 'best_D.mdl')
                    torch.save(dat.C_model.state_dict(), 'best_C.mdl')
                else:
                    print("nan error!!!")
                    dat.G_model.load_state_dict(torch.load('best.mdl'))
                    dat.D_model.load_state_dict(torch.load('best_D.mdl'))
                    dat.C_model.load_state_dict(torch.load('best_C.mdl'))
                    #return -1

            show_sum=show_sum+1

        #if True or epoch % 1==0 and epoch==0:
        #    val_acc=evalute(dat.model,dat.val_loader)
        #    if val_acc<best_acc:
        #        best_epoch=epoch
        #        best_acc=val_acc
        #        torch.save(dat.model.state_dict(),'best.mdl')
        #    print('val_acc:',val_acc,'best epoch:',best_epoch)

    print('best acc:',best_acc,'best epoch:',best_epoch)
    
    torch.save(dat.G_model.state_dict(),'best.mdl')
    #dat.model.load_state_dict(torch.load('best.mdl'))
    #print('loaded from ckpt!')
    
    test_loss=evalute(dat.G_model,dat.test_loader)
    print('test mse:',test_loss)

