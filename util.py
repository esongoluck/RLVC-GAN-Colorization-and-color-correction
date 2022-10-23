

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed
import config
import cv2
l_cent = 50.
l_norm = 100.
ab_norm = 128.

def cat_img(img1,img2):
	res=np.hstack((img1,img2.get()))
	return res

def uper_color(imga):
	aj = cv2.subtract(imga, 0.5)
	_, mask_aj = cv2.threshold(aj, 0, 2, cv2.THRESH_BINARY)
	mask_aj = cv2.subtract(mask_aj, 1.0)
	aj = cv2.pow(cv2.absdiff(aj, 0), 0.6)
	aj = cv2.multiply(aj, mask_aj)
	aj = cv2.add(aj, 0.5)
	return aj

def imwrite_tensor_lab(path,src):
	img = src.permute(1, 2, 0)
	img=img.cpu()
	img=img.float().numpy()
	img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)  ##########图像转化为BGR
	img=img*255
	cv2.imwrite(path,img)
def imwrite_tensor_l(path,src):
	img=src[0]
	img=img.cpu()
	img=img.float().numpy()
	img=img*255

	cv2.imwrite(path,img)

def yCbCr2rgb(input_im,device):
	im_flat = input_im.permute(1, 2, 0)
	mat = torch.tensor([[1.0, 1.0, 1.0],
						[1.40252454418, -0.7144034731, 0.0],
						[0.0, -0.344340135561, 1.77304964539]]).to(device).half()
	bias = torch.tensor([0.0,0.5,0.5]).to(device).half()
	temp = (im_flat - bias).matmul(mat)
	return temp.permute(2, 0, 1)


def rgb2yCbCr(input_im,device):
	im_flat = input_im.permute(1, 2, 0)
	mat = torch.tensor([[0.299, 0.499813, -0.168636],
						[0.587, -0.418531, -0.331068],
						[0.114, -0.01282, 0.499704]]).to(device).half()
	bias = torch.tensor([0.0,0.5,0.5]).to(device).half()
	temp = im_flat.matmul(mat) + bias
	return temp.permute(2, 0, 1)

def load_img(img_path):
	out_np = np.asarray(Image.open(img_path))
	if(out_np.ndim==2):
		out_np = np.tile(out_np[:,:,None],3)
	return out_np

def resize_img(img, HW=(256,256), resample=3):
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
	# return original size L and resized L as torch Tensors
	img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
	
	img_lab_orig = color.rgb2lab(img_rgb_orig)
	img_lab_rs = color.rgb2lab(img_rgb_rs)

	img_l_orig = img_lab_orig[:,:,0]
	img_l_rs = img_lab_rs[:,:,0]

	tens_orig_l = torch.Tensor(img_l_orig)[None,:,:]
	tens_rs_l = torch.Tensor(img_l_rs)[None,:,:]

	return (tens_orig_l, tens_rs_l)

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
	# tens_orig_l 	1 x H_orig x W_orig
	# out_ab 		2 x H x W

	HW_orig = tens_orig_l.shape[1:]
	HW = out_ab.shape[1:]
	print(tens_orig_l.shape,HW_orig,out_ab.shape,HW)
	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
	else:
		out_ab_orig = out_ab

	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=0)
	return color.lab2rgb(out_lab_orig.data.cpu().numpy().transpose((1,2,0)))#坐标轴xyz换为zxy并转rgb格式

def show_lab_images_32(labs):
	labs=labs.clone()
	labs=unnormalize_lab(labs)
	labs=labs.cpu().detach().numpy()
	labs=np.transpose(labs, (0,2,3,1))
	plt.figure(figsize=(labs.shape[3]*6, labs.shape[3]*6))
	for i in range(config.Config["batch_size"]):
		labsi=labs[i,:,:,:]
		labsi=color.lab2rgb(labsi)*255
		im = Image.fromarray(np.uint8(labsi))
		# if not os.path.exists("imgout/"):
		# 	os.mkdir("imgout/")
		im.save("imgout/test_imgout_"+str(i).zfill(5)+".png")
		plt.subplot(4,4,i+1)
		plt.imshow(labsi/255)
		plt.xticks([])
		plt.yticks([])
	plt.show()

def show_llab_images_32(labs):
	labs=labs.clone()
	labs=unnormalize_llab(labs)
	labs=labs.cpu().detach().numpy()
	labs=np.transpose(labs, (0,2,3,1))
	
	plt.figure(figsize=(4, 4))
	for i in range(config.Config["batch_size"]):

		labsi=labs[i,:,:,:]
		labi_L1=labs[i,:,:,0]
		labi_L2=labs[i,:,:,1]
		labi_L3=labs[i,:,:,2]
		labi_L4=labs[i,:,:,3]
		labsi=np.stack((labi_L2,labi_L3,labi_L4), axis=2)
		labsi=color.lab2rgb(labsi)*255

		plt.subplot(4,4,i+1)
		plt.imshow(labsi/255)
		plt.xticks([])
		plt.yticks([])

	plt.show()

	#plt.figure(figsize=(4, 4))
	#for i in range(config.Config["batch_size"]):
	#	labsi=labs[i,:,:,:]
	#	labi_L1=labs[i,:,:,0]
	#	labi_L2=labs[i,:,:,1]
	#	labi_L3=labs[i,:,:,2]
	#	labi_L4=labs[i,:,:,3]
	#	labsi=np.stack((labi_L1,0*labi_L1,0*labi_L1), axis=2)

	#	labsi2=color.lab2rgb(labsi)*255
	#	plt.subplot(4,4,i+1)
	#	plt.imshow(labsi2/255)
	#	plt.xticks([])
	#	plt.yticks([])
	#plt.show()
	



def normalize_lab(in_lab):
	#return in_lab
	lable_L = (in_lab[:,0,:,:]-l_cent)/l_norm*2
	lable_a = in_lab[:,1,:,:]/ab_norm
	lable_b = in_lab[:,2,:,:]/ab_norm
	return torch.stack([lable_L,lable_a,lable_b],dim=1)


def unnormalize_lab(in_lab):
	#return in_lab
	lable_L = in_lab[:,0,:,:]/2*l_norm+l_cent
	lable_a = (in_lab[:,1,:,:])*ab_norm
	lable_b = (in_lab[:,2,:,:])*ab_norm
	return torch.stack([lable_L,lable_a,lable_b],dim=1)

	

def normalize_llab(in_lab):
	#return in_lab
	lable_L0 = (in_lab[:,0,:,:]-l_cent)/l_norm*2
	lable_L = (in_lab[:,1,:,:]-l_cent)/l_norm*2
	lable_a = in_lab[:,2,:,:]/ab_norm
	lable_b = in_lab[:,3,:,:]/ab_norm
	return torch.stack([lable_L0,lable_L,lable_a,lable_b],dim=1)


def unnormalize_llab(in_lab):
	#return in_lab
	lable_L0 = in_lab[:,0,:,:]/2*l_norm+l_cent
	lable_L = in_lab[:,1,:,:]/2*l_norm+l_cent
	lable_a = (in_lab[:,2,:,:])*ab_norm
	lable_b = (in_lab[:,3,:,:])*ab_norm
	return torch.stack([lable_L0,lable_L,lable_a,lable_b],dim=1)


def normalize_l_3(in_l_3):
	#return in_l_3
	lable_L0 = (in_l_3[:,0,:,:]-l_cent)/l_norm*2
	lable_L1 = (in_l_3[:,1,:,:]-l_cent)/l_norm*2
	lable_L2 = (in_l_3[:,2,:,:]-l_cent)/l_norm*2
	return torch.stack([lable_L0,lable_L1,lable_L2],dim=1)

def unnormalize_l_3(in_l_3):
	#return in_l_3
	lable_L0 = in_l_3[:,0,:,:]/2*l_norm+l_cent
	lable_L1 = in_l_3[:,0,:,:]/2*l_norm+l_cent
	lable_L2 = in_l_3[:,0,:,:]/2*l_norm+l_cent
	return torch.stack([lable_L0,lable_L1,lable_L2],dim=1)

def normalize_lab_3(in_lab_3):
	#return in_lab_3
	lable_L0 = (in_lab_3[:,0,:,:]-l_cent)/l_norm*2
	lable_a0 = in_lab_3[:,1,:,:]/ab_norm
	lable_b0 = in_lab_3[:,2,:,:]/ab_norm
	lable_L1 = (in_lab_3[:,3,:,:]-l_cent)/l_norm*2
	lable_a1 = in_lab_3[:,4,:,:]/ab_norm
	lable_b1 = in_lab_3[:,5,:,:]/ab_norm
	lable_L2 = (in_lab_3[:,6,:,:]-l_cent)/l_norm*2
	lable_a2 = in_lab_3[:,7,:,:]/ab_norm
	lable_b2 = in_lab_3[:,8,:,:]/ab_norm
	return torch.stack([lable_L0,lable_a0,lable_b0,lable_L1,lable_a1,lable_b1,lable_L2,lable_a2,lable_b2],dim=1)


def unnormalize_lab_3(in_lab_3):
	#return in_lab_3
	lable_L0 = in_lab_3[:,0,:,:]/2*l_norm+l_cent
	lable_a0 = (in_lab_3[:,1,:,:])*ab_norm
	lable_b0 = (in_lab_3[:,2,:,:])*ab_norm
	lable_L1 = in_lab_3[:,3,:,:]/2*l_norm+l_cent
	lable_a1 = (in_lab_3[:,4,:,:])*ab_norm
	lable_b1 = (in_lab_3[:,5,:,:])*ab_norm
	lable_L2 = in_lab_3[:,6,:,:]/2*l_norm+l_cent
	lable_a2 = (in_lab_3[:,7,:,:])*ab_norm
	lable_b2 = (in_lab_3[:,8,:,:])*ab_norm
	return torch.stack([lable_L0,lable_a0,lable_b0,lable_L1,lable_a1,lable_b1,lable_L2,lable_a2,lable_b2,],dim=1)
