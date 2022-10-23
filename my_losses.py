import torch
import torch.nn as nn
import config

###############################################################       ssim
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp




def gaussian(window_size, sigma):
    #print("win_S",window_size)
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size)).to(torch.device(config.Config['device']))
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    paddings=int(window_size/2)
    mu1 = F.conv2d(img1, window, padding = paddings, groups = channel)
    mu2 = F.conv2d(img2, window, padding = paddings, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = paddings, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = paddings, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = paddings, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return torch.mean(ssim_map)
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Module):
    def __init__(self, window_size = 11,size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def __call__(self, in0, in1):
        (_, channel, _, _) = in0.size()
        self.window = create_window(self.window_size, channel)

        #if config.Config["half"] == "True":
            #self.window = self.window.half()
        return _ssim(in0, in1, self.window, self.window_size, channel, self.size_average)

# def ssim(img1, img2,window_size = 11, size_average = True):
#     (_, channel, _, _) = img1.size()
#     print("!!!!",channel)
#     window = create_window(window_size, channel)
#     return _ssim(img1, img2, window, window_size, channel, size_average)
###################################################################            ssim

class ssim_Loss(nn.Module):####################################################自定义loss
    def __init__(self):
        super(ssim_Loss, self).__init__()
        self.ssim_loss=SSIM()
    def __call__(self, in0, in1):
        return self.ssim_loss(in0,in1)


class mse_Loss(nn.Module):#################################################### mse loss
    def __init__(self):
        super(mse_Loss, self).__init__()
    def __call__(self, in0, in1):
        return torch.mean(torch.abs(in0-in1)**2)

class l1_Loss(nn.Module):#################################################### l1 loss
    def __init__(self):
        super(l1_Loss, self).__init__()
    def __call__(self, in0, in1):
        return torch.mean(torch.abs(in0-in1))

class psnr_Loss(nn.Module):####################################################自定义loss
    def __init__(self,MaxNum):
        super(psnr_Loss, self).__init__()
        self.MaxNum=MaxNum
        self.mse_loss=mse_Loss()
    def __call__(self, in0, in1):
        mse=self.mse_loss(in0,in1)
        min_mse=torch.tensor(0.001)
        if mse<min_mse:
            return 10*torch.log(self.MaxNum**2/min_mse)/torch.log(torch.tensor(10.0))
        return 10*torch.log(self.MaxNum**2/mse)/torch.log(torch.tensor(10.0))



class HuberLoss(nn.Module):####################################################自定义loss
    def __init__(self, delta=.01):
        super(HuberLoss, self).__init__()
        self.delta=delta

    def __call__(self, in0, in1):
        mask = torch.zeros_like(in0)
        mann = torch.abs(in0-in1)
        eucl = .5 * (mann**2)
        mask[...] = mann < self.delta

        # loss = eucl*mask + self.delta*(mann-.5*self.delta)*(1-mask)
        loss = eucl*mask/self.delta + (mann-.5*self.delta)*(1-mask)
        return torch.mean(torch.mean(loss,dim=1,keepdim=True))
    

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def __call__(self, in0, in1):
        return torch.mean(torch.abs(in0-in1))

class Gan_Loss(nn.Module):
    def __init__(self):
        super(Gan_Loss, self).__init__()

    def __call__(self, in0):
        
        if(config.Config["MD_sig"]==True):

            res=-torch.log(torch.max(1-torch.mean(in0),torch.tensor(0.001)))#虽然假，但是仍然要接近0
        else:
            res=torch.mean(torch.abs(in0))#虽然假，但是仍然要接近0
        return res

class direct_p0_Loss(nn.Module):#yi相比较
    def __init__(self):
        super(direct_p0_Loss, self).__init__()

    def __call__(self, in0, in1):#0是预测值，1是标签
        in1p=in1[:,6:9,:,:]
        res=torch.mean(torch.abs(in0-in1p)**2,dim=0,keepdim=False)
        res=torch.mean(res,dim=1,keepdim=False)
        res=torch.mean(res,dim=1,keepdim=False)
        return 0.5*res[0]+res[1]+res[2]#l   a   b
class direct_p1_Loss(nn.Module):#yi-1相比较
    def __init__(self):
        super(direct_p1_Loss, self).__init__()

    def __call__(self, in0, in1):
        in1p=in1[:,3:6,:,:]
        res=torch.mean(torch.abs(in0-in1p)**2,dim=0,keepdim=False)
        res=torch.mean(res,dim=1,keepdim=False)
        res=torch.mean(res,dim=1,keepdim=False)
        return 0.5*res[0]+res[1]+res[2]#l   a   b
class direct_p2_Loss(nn.Module):#yi-2相比较
    def __init__(self):
        super(direct_p2_Loss, self).__init__()

    def __call__(self, in0, in1):
        in1p=in1[:,0:3,:,:]
        res=torch.mean(torch.abs(in0-in1p)**2,dim=0,keepdim=False)
        res=torch.mean(res,dim=1,keepdim=False)
        res=torch.mean(res,dim=1,keepdim=False)
        return 0.5*res[0]+res[1]+res[2]#l   a   b


    
class direct_Loss(nn.Module):#y相比较
    def __init__(self):
        super(direct_Loss, self).__init__()
        self.direct_p0_Loss=direct_p0_Loss()
        self.direct_p1_Loss=direct_p1_Loss()
        self.direct_p2_Loss=direct_p2_Loss()
    def __call__(self, in0, in1):

        return self.direct_p0_Loss(in0, in1)+0.0*self.direct_p1_Loss(in0, in1)+0.0*self.direct_p2_Loss(in0, in1) #    0   -1   -2



class D_Loss(nn.Module):
    def __init__(self):
        super(D_Loss, self).__init__()

    def __call__(self, in0,in1):
        if(config.Config["MD_sig"]==True):
            res0=-torch.log(torch.max(1-torch.mean(in0),torch.tensor(0.0001)))#真的接近0
            res1=-torch.log(torch.max(torch.mean(in1),torch.tensor(0.0001)))#假的接近1
        else:
            res0=torch.mean(in0)#真的接近0
            res1=1-torch.mean(in1)#假的接近1
        return res0+res1


class C_Loss(nn.Module):
    def __init__(self):
        super(C_Loss, self).__init__()
        self.mse_loss=mse_Loss()
        #self.ssim_loss=ssim_Loss()
    def __call__(self, inx,inxp):
        return self.mse_loss(inx,inxp)

class color_Loss(nn.Module):
    def __init__(self):
        super(color_Loss, self).__init__()
        self.l1_loss=l1_Loss()
    def __call__(self, inyp):

        in_a=inyp[:,1:2,:,:]
        in_b=inyp[:,2:3,:,:]
        in_r=in_a*in_a+in_b*in_b
        in_r=torch.pow(in_r,0.5)
        return self.l1_loss(in_r,0.0)


class G_Loss(nn.Module):
    def __init__(self):
        super(G_Loss, self).__init__()
        self.Gan_Loss=Gan_Loss()
        self.direct_Loss=direct_Loss()
        self.ssim_loss=ssim_Loss()
        self.psnr_loss=psnr_Loss(1.0)
        self.C_loss=C_Loss()
        self.color_loss=color_Loss()
    def __call__(self, iny,inyp,inZp,x3,x_Cp):
        #print("loss_Direct",self.direct_Loss(inyp,iny),"loss_Gan",self.Gan_Loss(inZp))
        #print("loss:",self.direct_Loss(inyp,iny)+0.1*self.Gan_Loss(inZp)-self.ssim_loss(inyp,iny[:,6:9,:,:])-0.1*self.psnr_loss(inyp,iny[:,6:9,:,:])+0.1*self.C_loss(x3,x_Cp))
        return self.direct_Loss(inyp,iny)+self.Gan_Loss(inZp)-self.ssim_loss(inyp,iny[:,6:9,:,:])+self.C_loss(x3,x_Cp)-self.color_loss(inyp)
