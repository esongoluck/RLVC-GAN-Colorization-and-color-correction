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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def gaussian(window_size, sigma):
    #print("win_S",window_size)
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size)).to('cuda')
    return window


def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out

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

def _ssim2(X, Y, win, data_range=1023, size_average=True, full=False):
    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * ( gaussian_filter(X * X, win) - mu1_sq )
    sigma2_sq = compensation * ( gaussian_filter(Y * Y, win) - mu2_sq )
    sigma12   = compensation * ( gaussian_filter(X * Y, win) - mu1_mu2 )

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val

def ssim(in0,in1):
    window_size = 11
    size_average = True
    (_, channel, _, _) = in0.size()
    window = create_window(window_size, channel)
    return _ssim(in0, in1, window, window_size, channel, size_average)


def ms_ssim(X, Y, win_size=11, win_sigma=10, win=None, data_range=1, size_average=True, full=False, weights=None):
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if weights is None:
        weights = torch.FloatTensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(X.device, dtype=X.dtype)

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim2(X, Y,
                             win=win,
                             data_range=data_range,
                             size_average=False,
                             full=True)
        mcs.append(cs)

        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)
    # weights, (level)
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1))
                            * (ssim_val ** weights[-1]), dim=0)  # (batch, )

    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val

def ssim_Loss(in0,in1):
    return ssim(torch.tensor(in0).permute(2, 0, 1)[None,:,:,:].to('cuda'),torch.tensor(in1).permute(2, 0, 1)[None,:,:,:].to('cuda'))

def mssim_Loss(in0,in1):
    return ms_ssim(torch.tensor(in0).permute(2, 0, 1)[None,:,:,:].to('cuda'),torch.tensor(in1).permute(2, 0, 1)[None,:,:,:].to('cuda'))


def mse_Loss(in0,in1):#################################################### mse loss
    return torch.mean(torch.abs(torch.tensor(in0).permute(2, 0, 1)[None,:,:,:].to('cuda')-torch.tensor(in1).permute(2, 0, 1)[None,:,:,:].to('cuda'))**2)

def psnr_Loss(in0,in1):####################################################自定义loss
    mse=mse_Loss(in0,in1)
    MaxNum=1.0
    min_mse=torch.tensor(0.001)
    if mse<min_mse:
        return 10*torch.log(MaxNum**2/min_mse)/torch.log(torch.tensor(10.0))
    return 10*torch.log(MaxNum**2/mse)/torch.log(torch.tensor(10.0))

def cs_Loss(in0):
    in1 = cv2.cvtColor(in0, cv2.COLOR_BGR2HSV)
    return torch.tensor(cv2.mean(in1[:,:,1])[0]).to('cuda')



if __name__ == '__main__':
    irs_path = "result/our/ir"
    rgbs_path = "result/our/rgb"
    #rgb_falses_path = "result-dark/our/rgb_false"
    #rgb_finals_path = "result-dark/our/rgb_final"
    rgb_falses_path = "result/BIMEF"
    rgb_finals_path = "result/BIMEF"

    irs = glob.glob(os.path.join(irs_path, '*' + '.jpg'))
    rgbs = glob.glob(os.path.join(rgbs_path, '*' + '.jpg'))
    rgb_falses = glob.glob(os.path.join(rgb_falses_path, '*' + '.jpg'))
    rgb_finals = glob.glob(os.path.join(rgb_finals_path, '*' + '.jpg'))
    i=0
    sssim=0
    smssim=0
    spsnr=0
    smse=0
    scs=0
    for ir in irs:
        ir=irs[i]
        rgb=rgbs[i]
        rgb_false=rgb_falses[i]
        rgb_final=rgb_finals[i]
        ir=cv2.resize(cv2.imread(ir).astype(np.float32)/255,(512,512))
        rgb=cv2.resize(cv2.imread(rgb).astype(np.float32)/255,(512,512))
        rgb_false=cv2.resize(cv2.imread(rgb_false).astype(np.float32)/255,(512,512))
        rgb_final=cv2.resize(cv2.imread(rgb_final).astype(np.float32)/255,(512,512))




        ssim_our_i=ssim_Loss(rgb,rgb_false)
        mssim_our_i=mssim_Loss(rgb,rgb_false)
        psnr_our_i=psnr_Loss(rgb,rgb_false)
        mse_our_i=mse_Loss(rgb,rgb_false)
        cs_our_i=cs_Loss(rgb_final)





        sssim+=ssim_our_i
        smssim+=mssim_our_i
        spsnr+=psnr_our_i
        smse+=mse_our_i
        scs+=cs_our_i

        i+=1

    print(sssim/i," ",smssim/i," ",spsnr/i," ",smse/i," ",scs/i)