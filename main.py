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

if __name__ == '__main__':
    time_start = datetime.datetime.now()
    try:
        print('use gpu? ', torch.cuda.is_available(), '     gpu nums: ', torch.cuda.device_count())
        conf = config.Config
        dat = datas.datas()

        corners_file=open("cut_corners")#读取角点坐标
        lines=corners_file.readlines()
        i=0
        for line in lines :#逐行读文件

            x,_,y=line.partition(" ")
            x=x.replace('\n', '').replace(' ', '')
            y=y.replace('\n', '').replace(' ', '')
            if len(x)==0 or len(y)==0:
                continue
            if i%2==0:
                conf["x_corners"][int(i/2)][0],conf["x_corners"][int(i/2)][1]=float(x),float(y)
            else:
                conf["y_corners"][int(i/2)][0],conf["y_corners"][int(i/2)][1]=float(x),float(y)
            i=i+1
        conf["the_corners"][0][0],conf["the_corners"][0][1]=0,0                                         #生成变换矩阵
        conf["the_corners"][1][0],conf["the_corners"][1][1]=conf["image_size"]-1,0
        conf["the_corners"][2][0],conf["the_corners"][2][1]=0,conf["image_size"]-1
        conf["the_corners"][3][0],conf["the_corners"][3][1]=conf["image_size"]-1,conf["image_size"]-1
        conf["tr_mat_x"]=cv2.getPerspectiveTransform(conf["x_corners"],conf["the_corners"])
        conf["tr_mat_y"]=cv2.getPerspectiveTransform(conf["y_corners"],conf["the_corners"])


        if(conf["mood"]=="train"):
            train.to_train(dat,conf)


        image_size = int(conf["image_size"])

        model_size = int(conf["image_size"])        #模型大小


        dat.G_model=myGenerator()#.to(device)
        dat.G_model.load_state_dict(torch.load('best.mdl'), strict=True)
        #stat(dat.G_model.cpu(), (3, 256, 256))
        dat.G_model=dat.G_model.to(device)
        dat.G_model=dat.G_model.half()
        print("!!!",dat.G_model)
        dat.G_model.requires_grad_(False)
        ir_path = "my_dataset/test/img-ir/"
        rgb_path = "my_dataset/test/img-rgb/"

        irs = []
        rgbs = []  # 筛选后存这里(同时具备标签和红外图像的图像)
        names=[]
        type=".jpg"
        irs += glob.glob(os.path.join(ir_path, '*'+type))

        len_imgs=len(irs)
        for ir_pathi in irs:                                 #读取所有图像的名称
            name, _, typei = ir_pathi.rpartition('.')
            _, _, name = name.rpartition('/')
            _, _, name = name.rpartition('\\')
            names.append(name)
        rgb_non=np.zeros((conf["image_size"],conf["image_size"],3),dtype=np.float32)+0.5
        ir_non=np.zeros((model_size,model_size,1),dtype=np.float32)+0.5
        ir_non,rgb_non=cv2.UMat(ir_non),cv2.UMat(rgb_non)

        ir_white_block=np.zeros((model_size,2,1),dtype=np.float32)+1
        rgb_white_block=np.zeros((conf["image_size"],2,3),dtype=np.float32)+1
        ir_white_block,rgb_white_block=cv2.UMat(ir_white_block),cv2.UMat(rgb_white_block)

        ir_true_all,rgb_true_all,rgb_false_all,rgb_final_all=ir_white_block.get(),rgb_white_block.get(),rgb_white_block.get(),rgb_white_block.get()
        #cv2.imshow("wb",ir_white_block);
        #cv2.waitKey(0);
        ir,ir_1,ir_2,rgb_true,rgb_true_1,rgb_true_2=ir_non,ir_non,ir_non,rgb_non,rgb_non,rgb_non
        timei=time.time()
        sum=0

        # fig1,fig2,fig3,fig4=plt.figure(),plt.figure(),plt.figure(),plt.figure()
        # ax1,ax2,ax3,ax4=fig1.gca(),fig2.gca(),fig3.gca(),fig4.gca()
        # fig1.show(),fig2.show(),fig3.show(),fig4.show()

        time_start=time.time()
        bias = cv2.UMat(np.array([[255.0, 255.0, 255.0]]))
        #lpls=cv2.UMat(np.array([[0.0, 1.0, 0.0],[1.0, -4.0, 1.0],[0.0, 1.0, 0.0],]))
        #ir_in_now = torch.cat([ir_2, ir_1, ir], dim=0)[None, :, :, :]
        #ir_af,rgb_true_af=None,None
        maked_sum=0
        for name in names:
            sum=sum+1
            if not (23461-2<=int(name)<=23471):
                continue
            print("name:",int(name))
            maked_sum+=1
            if maked_sum==3:
                time_start=time.time()
            time0=time.time()
            ir_2=ir_1
            ir_1=ir
            rgb_true_2=rgb_true_1
            rgb_true_1=rgb_true

            ir_cpu=cv2.imread(ir_path+name+type).astype(np.float32)
            rgb_true_cpu=cv2.imread(rgb_path+name+type)
            rgb_true_cpu=rgb_true_cpu.astype(np.float32)

            time1=time.time()

            if rgb_true_cpu is None:
                rgb_true = rgb_non
            else:
                time_t0 = time.time()
                rgb_true = cv2.UMat(rgb_true_cpu)

                rgb_true = cv2.divide(rgb_true, bias)
                rgb_true = cv2.cvtColor(rgb_true, cv2.COLOR_BGR2YCrCb)

                rgb_true = cv2.warpPerspective(rgb_true, conf["tr_mat_y"], [image_size, image_size])

            #                ax2.imshow(transforms.ToPILImage()(rgb_true))
            #                fig2.canvas.draw()


            if ir_cpu is None:
                ir=ir_non
            else:
                ir=cv2.UMat(ir_cpu)
                ir=cv2.cvtColor(ir,cv2.COLOR_BGR2GRAY)
                ir=cv2.divide(ir,255.0)

                ################################# ir 向rgb对齐
                ir = cv2.warpPerspective(ir, conf["tr_mat_x"], [image_size, image_size])



            ir_in=cv2.merge([ir_2,ir_1,ir])
            time2=time.time()
            ir_in_tensor=torch.tensor(ir_in.get()).to(device).permute(2, 0, 1)[None,:,:,:].half()
            rgb_false_tensor=dat.G_model(ir_in_tensor)#.detach()
            #print(rgb_false_tensor)

            time3=time.time()

            rgb_false=cv2.UMat(rgb_false_tensor[0].permute(1, 2, 0).cpu().numpy().astype(np.float32))

            true_l,true_a,true_b=cv2.split(rgb_true)
            false_l,false_a,false_b=cv2.split(rgb_false)


            [minl,maxl,_,_]=cv2.minMaxLoc(true_l)#首先进行标准化，0-1定格
            true_l=cv2.divide(cv2.subtract(true_l,minl),maxl-minl)
            rgb_true3=cv2.merge([true_l,true_a,true_b])#要删掉


            higher_mask=cv2.blur(true_l,[20,20])#mask模糊化
            higher_mask_2=cv2.pow(higher_mask,0.5)
            higher_mask_2_div=cv2.subtract(1,higher_mask_2)
            higher_mask_3=cv2.pow(higher_mask,0.6)
            higher_mask_3_div=cv2.subtract(1,higher_mask_3)
            # print("!!!",higher_mask.get())
            # print("!!!",higher_mask_2.get())
            # print("!!!",higher_mask_2_div.get())

            #higher_mask=cv2.multiply(higher_mask,10)
            #_,higher_mask_2=cv2.threshold(higher_mask,0.1,1,cv2.THRESH_TOZERO)#mask暗区归零

            final_l=cv2.add(cv2.multiply(true_l,higher_mask_2),cv2.multiply(false_l,higher_mask_2_div))
            final_a=cv2.add(cv2.multiply(true_a,higher_mask_3),cv2.multiply(false_a,higher_mask_3_div))
            final_b=cv2.add(cv2.multiply(true_b,higher_mask_3),cv2.multiply(false_b,higher_mask_3_div))
            rgb_final=cv2.merge([final_l,final_a,final_b])



            # true_l_cpu=cv2.multiply(true_l,255*255).get().astype(np.uint16)#均值化
            # clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(1, 1))
            # true_l_cpu = clahe.apply(true_l_cpu).astype(np.float32)
            # true_l=cv2.divide(cv2.UMat(true_l_cpu),255*255)
            # rgb_true2=cv2.merge([true_l,true_a,true_b])
            rgb_true2 = rgb_true

            rgb_false = cv2.cvtColor(rgb_false, cv2.COLOR_YCrCb2BGR)              ##########图像转化为BGR
            rgb_true = cv2.cvtColor(rgb_true, cv2.COLOR_YCrCb2BGR)              ##########图像转化为BGR
            #rgb_true2 = cv2.cvtColor(rgb_true2, cv2.COLOR_YCrCb2BGR)              ##########图像转化为BGR
            #rgb_true3 = cv2.cvtColor(rgb_true3, cv2.COLOR_YCrCb2BGR)              ##########图像转化为BGR
            rgb_final = cv2.cvtColor(rgb_final, cv2.COLOR_YCrCb2BGR)              ##########图像转化为BGR
#            ax3.imshow(transforms.ToPILImage()(rgb_false))
#            fig3.canvas.draw()


            time4=time.time()
#
            cv2.imwrite("imgout/rgb_final.jpg",cv2.multiply(rgb_final,bias))
#             cv2.imwrite("imgout/rgb_false.jpg",cv2.multiply(rgb_false,bias))
#             cv2.imwrite("imgout/ir_true.jpg",cv2.multiply(ir,255))
#             cv2.imwrite("imgout/rgb_true.jpg",cv2.multiply(rgb_true,bias))
#             cv2.imwrite("imgout/ir/"+str(int(name)).zfill(6)+".jpg",cv2.multiply(ir,255))
#             cv2.imwrite("imgout/rgb/"+str(int(name)).zfill(6)+".jpg",cv2.multiply(rgb_true,bias))
#             cv2.imwrite("imgout/rgb_false/"+str(int(name)).zfill(6)+".jpg",cv2.multiply(rgb_false,bias))
#             cv2.imwrite("imgout/rgb_final/"+str(int(name)).zfill(6)+".jpg",cv2.multiply(rgb_final,bias))
#
#
#             # cv2.imshow("ir",ir)
#             # cv2.imshow("rgb_true",rgb_true)
#             # cv2.imshow("rgb_false",rgb_false)
#             # cv2.imshow("final",rgb_final)
#             # cv2.waitKey(10)
#             cv2.imwrite("imgout/rgb_true2.jpg",cv2.multiply(rgb_true2,bias))
#             cv2.imwrite("imgout/rgb_true3.jpg",cv2.multiply(rgb_true3,bias))
#             cv2.imwrite("imgout/higher_mask.jpg",cv2.multiply(higher_mask,255))
#             cv2.imwrite("imgout/higher_mask_2.jpg",cv2.multiply(higher_mask_2,255))
#
#             if maked_sum>2:
#                 print("cat!!!!!!!!!!!!")
#                 ir_true_all=util.cat_img(ir_true_all,ir)
#                 ir_true_all = util.cat_img(ir_true_all, ir_white_block)
#
#                 rgb_true_all=util.cat_img(rgb_true_all,rgb_true)
#                 rgb_true_all = util.cat_img(rgb_true_all, rgb_white_block)
#
#                 rgb_false_all=util.cat_img(rgb_false_all,rgb_false)
#                 rgb_false_all = util.cat_img(rgb_false_all, rgb_white_block)
#
#                 rgb_final_all=util.cat_img(rgb_final_all,rgb_final)
#                 rgb_final_all = util.cat_img(rgb_final_all, rgb_white_block)
#
#
#
#
# #            cv2.waitKey(10)
#
#             cv2.imwrite("imgout/ir_true_all.jpg",cv2.multiply(ir_true_all,255))
#             cv2.imwrite("imgout/rgb_true_all.jpg",cv2.multiply(rgb_true_all,bias.get()))
#             cv2.imwrite("imgout/rgb_false_all.jpg",cv2.multiply(rgb_false_all,bias.get()))
#             cv2.imwrite("imgout/rgb_final_all.jpg",cv2.multiply(rgb_final_all,bias.get()))
#             time5=time.time()
            print("timei:",time1-time0,time2-time1,time3-time2,time4-time3,time4-timei,(time4-time_start)/max(1,maked_sum-2))
            timei=time.time()
            #cc=input("continue?")

        print("!!!!!!!!!!  right !!!!!!!!!!")
        #os.system("shutdown")
        #sys.exit(0)
    except :
        print("!!!!!!!!!!  error  !!!!!!!!!!")
        traceback.print_exc()
        #os.system("shutdown")
        #sys.exit(-1)
    time_end = datetime.datetime.now()
    print("End running time: " , time_end)
    print("Running time: ",(time_end-time_start))

