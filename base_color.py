

import torch
from torch import nn

class BaseColor(nn.Module):
	def __init__(self):
		super(BaseColor, self).__init__()

		self.l_cent = 50.
		self.l_norm = 100.
		self.ab_norm = 128.

	def normalize_l(self, in_l):
		return (in_l)/self.l_norm

	def unnormalize_l(self, in_l):
		return in_l*self.l_norm

	def normalize_ab(self, in_ab):
		return in_ab/self.ab_norm/2+0.5

	def unnormalize_ab(self, in_ab):
		return (in_ab-0.5)*2*self.ab_norm
	

	def normalize_lab(self, in_lab):
		lable_L = in_lab[:,0,:,:]/self.l_norm
		lable_a = in_lab[:,1,:,:]/self.ab_norm/2+0.5
		lable_b = in_lab[:,2,:,:]/self.ab_norm/2+0.5
		return torch.stack([lable_L,lable_a,lable_b],dim=1)


	def unnormalize_lab(self, in_lab):
		lable_L = in_lab[:,0,:,:]*self.l_norm
		lable_a = (in_lab[:,1,:,:]-0.5)*2*self.ab_norm
		lable_b = (in_lab[:,2,:,:]-0.5)*2*self.ab_norm
		return torch.stack([lable_L,lable_a,lable_b],dim=1)

	

	def normalize_llab(self, in_lab):
		lable_L0 = in_lab[:,0,:,:]/self.l_norm
		lable_L = in_lab[:,1,:,:]/self.l_norm
		lable_a = in_lab[:,2,:,:]/self.ab_norm/2+0.5
		lable_b = in_lab[:,3,:,:]/self.ab_norm/2+0.5
		return torch.stack([lable_L0,lable_L,lable_a,lable_b],dim=1)


	def unnormalize_llab(self, in_lab):
		lable_L0 = in_lab[:,0,:,:]*self.l_norm
		lable_L = in_lab[:,1,:,:]*self.l_norm
		lable_a = (in_lab[:,2,:,:]-0.5)*2*self.ab_norm
		lable_b = (in_lab[:,3,:,:]-0.5)*2*self.ab_norm
		return torch.stack([lable_L0,lable_L,lable_a,lable_b],dim=1)

