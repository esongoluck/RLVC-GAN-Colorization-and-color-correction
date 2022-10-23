
import torch
import torch.nn as nn
from base_color import *
import config

def base_block(start_nums,end_nums):
    #mid_nums=torch.max(torch.tensor([int(start_nums/4),3]))
    mid_nums=max(int(min(start_nums,end_nums)/4),8)
    block=[nn.Conv2d(start_nums, mid_nums, kernel_size=1, stride=1, padding=0, bias=True),]
    block+=[nn.LeakyReLU(True),]
    block+=[nn.Conv2d(mid_nums, mid_nums, kernel_size=3, stride=1, padding=1, bias=True),]
    block+=[nn.LeakyReLU(True),]
    block+=[nn.Conv2d(mid_nums, mid_nums, kernel_size=3, stride=1, padding=1, bias=True),]
    block+=[nn.LeakyReLU(True),]
    block+=[nn.Conv2d(mid_nums, mid_nums, kernel_size=3, stride=1, padding=1, bias=True),]
    block+=[nn.LeakyReLU(True),]
    block+=[nn.Conv2d(mid_nums, end_nums, kernel_size=1, stride=1, padding=0, bias=True),]
    block+=[nn.LeakyReLU(True),]
    block+=[nn.BatchNorm2d(end_nums),]
    return block


def in_block(start_nums,end_nums):
    #mid_nums=torch.max(torch.tensor([int(start_nums/4),3]))
    mid_nums=max(int(end_nums/4),8)
    block=[nn.Conv2d(start_nums, mid_nums, kernel_size=1, stride=1, padding=0, bias=True),]
    block+=[nn.LeakyReLU(True),]
    block+=[nn.Conv2d(mid_nums, mid_nums, kernel_size=3, stride=1, padding=1, bias=True),]
    block+=[nn.LeakyReLU(True),]
    block+=[nn.Conv2d(mid_nums, mid_nums, kernel_size=3, stride=1, padding=1, bias=True),]
    block+=[nn.LeakyReLU(True),]
    block+=[nn.Conv2d(mid_nums, mid_nums, kernel_size=3, stride=1, padding=1, bias=True),]
    block+=[nn.LeakyReLU(True),]
    block+=[nn.Conv2d(mid_nums, end_nums, kernel_size=1, stride=1, padding=0, bias=True),]
    block+=[nn.LeakyReLU(True),]
    return block

def out_block(start_nums,end_nums):
    #mid_nums=torch.max(torch.tensor([int(start_nums/4),3]))
    mid_nums=max(int(start_nums/4),8)
    block=[nn.Conv2d(start_nums, mid_nums, kernel_size=1, stride=1, padding=0, bias=True),]
    block+=[nn.LeakyReLU(True),]
    block+=[nn.Conv2d(mid_nums, mid_nums, kernel_size=3, stride=1, padding=1, bias=True),]
    block+=[nn.LeakyReLU(True),]
    block+=[nn.Conv2d(mid_nums, end_nums, kernel_size=1, stride=1, padding=0, bias=True),]
    block+=[nn.LeakyReLU(True),]
    return block


def base_block_up(start_nums,end_nums):

    #mid_nums=torch.max(torch.tensor([int(start_nums/4),3]))
    mid_nums=max(int(min(start_nums,end_nums)/4),8)
    block=[nn.Conv2d(start_nums, mid_nums, kernel_size=1, stride=1, padding=0, bias=True),]
    block+=[nn.LeakyReLU(True),]
    block+=[nn.ConvTranspose2d(mid_nums, mid_nums, kernel_size=4, stride=2, padding=1, bias=True), ]
    block+=[nn.LeakyReLU(True),]
    block+=[nn.Conv2d(mid_nums, end_nums, kernel_size=1, stride=1, padding=0, bias=True),]
    block+=[nn.LeakyReLU(True),]
    block+=[nn.BatchNorm2d(end_nums),]
    # block=[nn.ConvTranspose2d(start_nums, end_nums, kernel_size=4, stride=2, padding=1, bias=True), ]
    # block+=[nn.LeakyReLU(True),]
    # block+=[nn.BatchNorm2d(end_nums),]
    return block


class myGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d, in_nums=3,out_nums=3,F_nums = [128,256,512, 512],use_sigmoid=False):
        super(myGenerator, self).__init__()
        self.in_nums=in_nums
        self.out_nums=out_nums
        self.F_nums=F_nums

        sigmoid = [nn.Sigmoid(), ]
        # model_out+=[nn.Tanh()]
        #############################################

        self.M_l0=nn.Sequential(*base_block(in_nums, F_nums[0]))#这里不能用for循环，否则使用半精度加速时会出现未知原因模型无法切换为半精度
        self.M_l1=nn.Sequential(*base_block(F_nums[0], F_nums[1]))
        self.M_l2=nn.Sequential(*base_block(F_nums[1], F_nums[2]))
        self.M_l3=nn.Sequential(*base_block(F_nums[2], F_nums[3]))

        self.M_r2=nn.Sequential(*base_block_up(F_nums[3], F_nums[2]))
        self.M_r1=nn.Sequential(*base_block_up(F_nums[2], F_nums[1]))
        self.M_r0=nn.Sequential(*base_block_up(F_nums[1], F_nums[0]))#这里不能用for循环，否则使用半精度加速时会出现未知原因模型无法切换为半精度


        self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4), ])
        self.upsample05 = nn.Sequential(*[nn.Upsample(scale_factor=0.5), ])
        self.sigmoid = nn.Sequential(*sigmoid)
        self.softmax = nn.Sequential(*[nn.Softmax(dim=1), ])

        if use_sigmoid:
            #sequence +=
            self.M_rout = nn.Sequential(*(out_block(F_nums[0], out_nums)+sigmoid))
        else:
            self.M_rout = nn.Sequential(*out_block(F_nums[0], out_nums))
    def forward(self, input_A):
        L0=self.M_l0(input_A)
        L1=self.M_l1(self.upsample05(L0))
        L2=self.M_l2(self.upsample05(L1))
        L3=self.M_l3(self.upsample05(L2))
        R3=L3#暂时这样，底部省略
        R2=self.M_r2(R3)
        R1=self.M_r1(L2+R2)
        R0=self.M_r0(L1+R1)
        R_out=self.M_rout(L0+R0)
        #print("R_out",R_out.shape,R_out)


        return R_out
        #return self.unnormalize_lab(out_reg)

class SIGGRAPHGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d, in_nums=3, out_nums=3):
        super(SIGGRAPHGenerator, self).__init__()

        start_nums = 512
        end_nums = 256
        model8up = [nn.ConvTranspose2d(start_nums, end_nums, kernel_size=4, stride=2, padding=1, bias=True)]
        model3short8 = [nn.Conv2d(end_nums, end_nums, kernel_size=3, stride=1, padding=1, bias=True), ]

        model8 = [nn.ReLU(True), ]
        model8 += [nn.Conv2d(end_nums, end_nums, kernel_size=3, stride=1, padding=1, bias=True), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.Conv2d(end_nums, end_nums, kernel_size=3, stride=1, padding=1, bias=True), ]
        model8 += [nn.ReLU(True), ]
        model8 += [norm_layer(end_nums), ]

        # Conv9
        start_nums = 256
        end_nums = 128
        model9up = [nn.ConvTranspose2d(start_nums, end_nums, kernel_size=4, stride=2, padding=1, bias=True), ]
        model2short9 = [nn.Conv2d(end_nums, end_nums, kernel_size=3, stride=1, padding=1, bias=True), ]
        # add the two feature maps above

        model9 = [nn.ReLU(True), ]
        model9 += [nn.Conv2d(end_nums, end_nums, kernel_size=3, stride=1, padding=1, bias=True), ]
        model9 += [nn.ReLU(True), ]
        model9 += [norm_layer(end_nums), ]

        # Conv10
        model10up = [nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True), ]
        model1short10 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True), ]
        # add the two feature maps above

        model10 = [nn.ReLU(True), ]
        model10 += [nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=True), ]
        model10 += [nn.LeakyReLU(negative_slope=.2), ]

        # regression output
        model_out = [nn.Conv2d(128, 8, kernel_size=1, padding=0, dilation=1, stride=1, bias=True), ]
        model_out += [nn.Conv2d(8, out_nums, kernel_size=1, padding=0, dilation=1, stride=1, bias=True), ]
        sigmoid = [nn.Sigmoid(), ]
        # model_out+=[nn.Tanh()]
        #############################################
        self.model1 = nn.Sequential(*base_block(in_nums, 64))
        self.model2 = nn.Sequential(*base_block(64, 128))
        self.model3 = nn.Sequential(*base_block(128, 256))
        self.model4 = nn.Sequential(*base_block(256, 512))
        self.model5 = nn.Sequential(*base_block(512, 512))
        self.model6 = nn.Sequential(*base_block(512, 512))
        self.model7 = nn.Sequential(*base_block(512, 512))
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)

        self.model_out = nn.Sequential(*model_out)

        self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4, mode='bilinear'), ])
        self.sigmoid = nn.Sequential(*sigmoid)
        self.softmax = nn.Sequential(*[nn.Softmax(dim=1), ])
        self.upsample05 = nn.Sequential(*[nn.Upsample(scale_factor=0.5), ])

    def forward(self, input_A):
        # if(input_B is None):
        #    input_B = 0
        # if(mask_B is None):
        #    mask_B = 0

        # conv1_2 = self.model1(torch.cat((self.normalize_l(input_A),self.normalize_ab(input_B),mask_B),dim=1))

        conv1_2 = self.model1(input_A)
        conv2_2 = self.model2(self.upsample05(conv1_2))
        conv3_3 = self.model3(self.upsample05(conv2_2))
        conv4_3 = self.model4(self.upsample05(conv3_3))


        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)

        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)
        out_reg = self.model_out(conv10_2)

        return out_reg
        # return self.unnormalize_lab(out_reg)

class SIGGRAPHGenerator_1(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(SIGGRAPHGenerator_1, self).__init__()

        start_nums=6
        end_nums=16
        model1=[nn.Conv2d(start_nums, end_nums, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.LeakyReLU(True),]
        model1+=[nn.Conv2d(end_nums, end_nums, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[nn.LeakyReLU(True),]
        model1+=[norm_layer(end_nums),]

        model2=[nn.Conv2d(16, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.LeakyReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.LeakyReLU(True),]
        model2+=[norm_layer(128),]

        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.LeakyReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.LeakyReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[nn.LeakyReLU(True),]
        model3+=[norm_layer(256),]

        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.LeakyReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.LeakyReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.LeakyReLU(True),]
        model4+=[norm_layer(512),]

        model5=[nn.Conv2d(512, 128, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.LeakyReLU(True),]
        model5+=[nn.Conv2d(128, 128, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.LeakyReLU(True),]
        model5+=[nn.Conv2d(128, 128, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.LeakyReLU(True),]
        model5+=[norm_layer(128),]

        model6=[nn.Conv2d(128, 64, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.LeakyReLU(True),]
        model6+=[nn.Conv2d(64, 64, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.LeakyReLU(True),]
        model6+=[nn.Conv2d(64, 64, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.LeakyReLU(True),]
        model6+=[norm_layer(64),]

        model7=[nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.LeakyReLU(True),]
        model7+=[nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.LeakyReLU(True),]
        model7+=[nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.LeakyReLU(True),]
        model7+=[norm_layer(32),]

        model8=[nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[nn.LeakyReLU(True),]
        model8+=[nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.LeakyReLU(True),]
        model8+=[nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.LeakyReLU(True),]

        model8+=[nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0, bias=True),]
        model_out=[nn.Conv2d(8, 1, kernel_size=1, padding=0, stride=1, bias=False),]
        sigmoid=[nn.Sigmoid(),]
        upsample1=[nn.Upsample(1),]
        #############################################

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)
        
        self.model_out=nn.Sequential(*model_out)
        self.sigmoid=nn.Sequential(*sigmoid)
        self.upsample1=nn.Sequential(*upsample1)
        



    def forward(self, input_A):
        #inputs=torch.cat((self.normalize_l(input_A),self.normalize_ab(input_B)),dim=1)
        conv1_2 = self.model1(input_A)
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(conv8_3)
        upt=self.upsample1(out_reg)
        #print("upt",upt)
        sig=self.sigmoid(out_reg)

        return sig
        #return self.unnormalize_lab(out_reg)
        


def Rese_Blocks(start_nums,end_nums):
    squence=[]
    squence+=[]
    return squence


class NLayerGenerator(BaseColor):
    def __init__(self, input_nums, output_nums,ndf=8, n_layers=8, norm_layer=nn.BatchNorm2d,use_bias=False,use_sigmoid=False):
        super(NLayerGenerator, self).__init__()

        kw = 3
        st=1
        padw = 1

        nf_mult = 1
        nf_mult_prev = 1
        
        sequence = [
            nn.Conv2d(input_nums, input_nums*ndf, kernel_size=kw, stride=st, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(input_nums*ndf * nf_mult_prev, input_nums*ndf * nf_mult,
                          kernel_size=kw, stride=st, padding=padw, bias=use_bias),
                norm_layer(input_nums*ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(input_nums*ndf * nf_mult_prev, input_nums*ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(input_nums*ndf * nf_mult),
            nn.LeakyReLU( True)

        ]

        sequence += [nn.Conv2d(input_nums*ndf * nf_mult, output_nums, kernel_size=kw, stride=1, padding=padw)]
        print("sig:",use_sigmoid)
        if use_sigmoid is True:
            sequence += [nn.Sigmoid()]
            print("sig!!!")

        self.model = nn.Sequential(*sequence)

    def forward(self, input_A):
        return self.model(input_A)

class NLayerDiscriminator(BaseColor):
    def __init__(self, input_nums,output_nums, ndf=16, n_layers=3, norm_layer=nn.BatchNorm2d,use_bias=True,use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()

        kw = 3
        st=1
        padw = 1
        sequence = [
            nn.Conv2d(input_nums, ndf, kernel_size=kw, stride=st, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=st, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input_A):
        return self.model(input_A)
