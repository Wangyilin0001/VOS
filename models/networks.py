import cv2
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from models.init_weights import init_weights

from init_weights import init_weights
import math
from attention import PAAttention
from util import util
from torchvision.ops import DeformConv2d


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


# (输入图像通道-3，输出图像通道-3，第一个卷积层的生成过滤器（层数）-32，选择那个模型作为生成器-MSUNet，选择归一化方法-instance，是否使用dropout层—使用，网络初始化-normal（正态分布初始化）,设备号-0)
def define_G1(input_nc, output_nc, ngf, which_model_netG1, norm='batch', use_dropout=False, init_type='normal',
             gpu_ids=[]):
    netG1 = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG1 == 'resnet_9blocks':
        netG1 = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG1 == 'resnet_6blocks':
        netG1 = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG1 == 'unet_128':
        netG1 = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG1 == 'MSUNet_G1':
        netG1 = MSUNet()
    elif which_model_netG1 == 'LK_GAN_G1':
        netG1 = LKAT_network()
    elif which_model_netG1 == 'TICC_GAN_G1':
        netG1 = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
                             n_blocks_local=3, norm_layer=norm_layer)
    elif which_model_netG1 == 'unet_256':
        netG1 = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG1)
    return init_net(netG1, init_type, gpu_ids)

def define_G2(input_nc, output_nc, ngf, which_model_netG2, norm='batch', use_dropout=False, init_type='normal',
             gpu_ids=[]):
    netG2 = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG2 == 'resnet_9blocks':
        netG2 = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG2 == 'resnet_6blocks':
        netG2 = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG2 == 'unet_128':
        netG2 = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG2 == 'MSUNet_G2':
        netG2 = MSUNet()
    elif which_model_netG2 == 'LK_GAN_G2':
        netG2 = LKAT_network()
    elif which_model_netG2 == 'TICC_GAN_G2':
        netG2 = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
                          n_blocks_local=3, norm_layer=norm_layer)
    elif which_model_netG2 == 'unet_256':
        netG2 = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG2)
    return init_net(netG2, init_type, gpu_ids)


def define_D2(input_nc, ndf, which_model_netD2,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD2 = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD2 == 'basic_D2':
        netD2 = NLayerDiscriminator_D2(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD2 == 'n_layers':
        netD2 = NLayerDiscriminator_D2(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD2 == 'pixel':
        netD2 = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD2 == 'multi':
        netD2 = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D=3, getIntermFeat=False)
    elif which_model_netD2 == 'LK_GAN_D2':
        netD2 = NLayerDiscriminator_LK(input_nc)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD2)
    return init_net(netD2, init_type, gpu_ids)


def define_D1(input_nc, ndf, which_model_netD1,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD1 = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD1 == 'basic_D1':
        netD1 = NLayerDiscriminator_D1(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD1 == 'n_layers':
        netD1 = NLayerDiscriminator_D1(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD1 == 'pixel':
        netD1 = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD1 == 'multi':
        netD1 = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D=3, getIntermFeat=False)
    elif which_model_netD1 == 'LK_GAN_D1':
        netD1 = NLayerDiscriminator_LK(input_nc)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD1)
    return init_net(netD1, init_type, gpu_ids)

##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label)) # 在PyTorch模型中注册一个缓冲区,用于存储真实样本的标签
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()  # L1损失
        else:
            self.loss = nn.BCEWithLogitsLoss()  # 用  loss = -target * log(sigmoid(input)) - (1 - target) * log(1 - sigmoid(input))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:                                          # D_fake
            target_tensor = self.fake_label            # target_tensor = tensor(0)
        return target_tensor.expand_as(input)          # 对target_tensor进行扩展，相当于input加一个标签  input：0

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class GANLoss_multi(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss_multi, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

    #     # 等价于
    #
    #     # 分析：两层下采样结构太少了
    #
    #     input_R2d = nn.ReflectionPad2d(3)
    #     input_Conv = nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,bias=use_bias)
    #     input_BN = norm_layer(ngf)
    #     int_Re = nn.ReLU(True)
    #
    #     self.X00 = nn.Sequential(input_R2d,input_Conv,input_BN,int_Re)
    #
    #     # 三次下采样
    #     Conv_X01 =  nn.Conv2d(ngf, ngf* 2, kernel_size=3,stride=2, padding=1, bias=use_bias)
    #     BN_X01 =  nn.BatchNorm2d(ngf * 2)
    #     Re_X01 =  nn.ReLU(True)
    #
    #     self.X01 = nn.Sequential(Conv_X01,BN_X01,Re_X01)
    #
    #     Conv_X02 =  nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3,stride=2, padding=1, bias=use_bias)
    #     BN_X02 =  nn.BatchNorm2d(ngf * 4)
    #     Re_X02 =  nn.ReLU(True)
    #
    #     self.X02 = nn.Sequential(Conv_X02, BN_X02, Re_X02)
    #
    #     # Conv_X03 =  nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3,stride=2, padding=1, bias=use_bias)
    #     # BN_X03 =  nn.BatchNorm2d(ngf * 8)
    #     # Re_X03 =  nn.ReLU(True)
    #
    #     # 9次 ResnetBlock
    #     Resnet = []
    #     for i in range(n_blocks):
    #         Resnet += [ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
    #                               use_bias=use_bias)]
    #
    #     self.Res = nn.Sequential(*Resnet)
    #
    #     Conv_Y01 = nn.ConvTranspose2d(ngf * 4, int(ngf * 2),kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias)
    #     BN_Y01 =  nn.BatchNorm2d(ngf * 2)
    #     Re_Y01 =  nn.ReLU(True)
    #
    #     self.Y01 =nn.Sequential(Conv_Y01, BN_Y01, Re_Y01)
    #
    #     Conv_Y00 = nn.ConvTranspose2d(ngf * 2, int( ngf * 1 ),kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias)
    #     BN_Y00 =  nn.BatchNorm2d(ngf * 1)
    #     Re_Y00 =  nn.ReLU(True)
    #
    #     self.Y00 = nn.Sequential(Conv_Y00, BN_Y00, Re_Y00)
    #
    #     # Conv_Y01 = nn.ConvTranspose2d(ngf * 1, int( ngf / 2 ),kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias)
    #     # BN_Y01 =  nn.BatchNorm2d(ngf / 2)
    #     # Re_Y01 =  nn.ReLU(True)
    #
    #     out_R2d = nn.ReflectionPad2d(3)
    #     out_Conv = nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
    #     out_Tanh = nn.Tanh()
    #
    #     self.out = nn.Sequential(out_R2d, out_Conv, out_Tanh)
    #
    #
    # def forward(self, input):
    #     X00 = self.X00(input)
    #     X01 = self.X01(X00)
    #     X02 = self.X02(X01)
    #     X02_Res = self.Res(X02)
    #     Y01 = self.Y01(X02_Res)
    #     Y00 = self.Y00(Y01)
    #
    #     out = self.out(Y00)
    #
    #
    #     return out
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2

        for i in range(n_downsampling):   # 包含 0，1,2
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling   # 4
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)



class ResnetBlock_gll(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock_gll, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator_D1(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator_D1, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model_D1 = nn.Sequential(*sequence)

    def forward(self, input):
        # print("NlayerDis")
        return self.model_D1(input)


class NLayerDiscriminator_D2(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator_D2, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model_D2 = nn.Sequential(*sequence)

    def forward(self, input):
        # print("NlayerDis")
        return self.model_D2(input)


class NLayerDiscriminator_multi(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator_multi, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator_multi(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


##cascaded network
class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-12, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):

        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x

class unetConv2_2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2_2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            if hasattr(torch.cuda,'empty_cache'):
                torch.cuda.empty_cache()
            x = conv(x)
        return x

class unetConv2_com(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2_com, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        self.com = PAAttention(channel=out_size, reduction=8,kernel_size=7)

        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            if hasattr(torch.cuda,'empty_cache'):
                torch.cuda.empty_cache()
            x = conv(x)
        x = self.com(x)
        return x

class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin, self).__init__()
        if is_deconv:
            self.conv = unetConv2_2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConv2_2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)

class unetUp_origin_end(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin_end, self).__init__()
        if is_deconv:
            self.conv = unetConv2_com(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConv2_com(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)



class TRBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(TRBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.upmid = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=2, bias=False,dilation=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.downmid = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=2, bias=False,dilation=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.dila = nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,padding=1,dilation=1)
        self.final = nn.Conv2d(in_channels=3*out_ch,out_channels=out_ch,kernel_size=1,stride=1)
        self.right = shortcut

    def forward(self, x):
        up1 = self.up(x)
        up2 = self.upmid(up1)
        up3 = self.upmid(up2)
        up3 = up3 + up1

        down1 = self.down(x)
        down2 = self.downmid(down1)
        down3 = self.downmid(down2)

        mid = up2 * down2
        mid = self.dila(mid)
        out = torch.cat((up3, down3,mid), 1)

        out = self.final(out)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class MSUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=3, feature_scale=4, is_deconv=True, is_batchnorm=True, is_ds=True):
        super(MSUNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.feature_scale = feature_scale
        self.CatChannels = 64
        filters = [32, 64, 128, 256, 512]

        # downsampling
        self.conv00 = self.make_layer(self.in_channels, filters[0], 2)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = self.make_layer(filters[0], filters[1], 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = self.make_layer(filters[1], filters[2], 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = self.make_layer(filters[2], filters[3], 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = self.make_layer(filters[3], filters[4], 2)

        # upsampling
        self.up_concat01 = unetUp_origin(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp_origin(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp_origin(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp_origin_end(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp_origin(filters[1], filters[0], self.is_deconv, 2)
        self.up_concat12 = unetUp_origin(filters[2], filters[1], self.is_deconv, 2)
        self.up_concat22 = unetUp_origin_end(filters[3], filters[2], self.is_deconv, 2)

        self.up_concat03 = unetUp_origin(filters[1], filters[0], self.is_deconv, 2)
        self.up_concat13 = unetUp_origin_end(filters[2], filters[1], self.is_deconv, 2)

        self.up_concat04 = unetUp_origin_end(filters[1], filters[0], self.is_deconv, 2)

        #upsampling_add
        #layer4

        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)


        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # self.conv31 = nn.Conv2d(448, 256, 1)
        self.conv31 = nn.Sequential(nn.Conv2d(448, 256, 1),
                             nn.BatchNorm2d(256),
                             nn.ReLU(inplace=True), )

        # layer3
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1],self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(filters[1])
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # self.conv22 = nn.Conv2d(256, 128, 1)
        self.conv22 = nn.Sequential(nn.Conv2d(256, 128, 1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True), )

        # layer2
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # self.conv13 = nn.Conv2d(128, 64, 1)
        self.conv13 = nn.Sequential(nn.Conv2d(128, 64, 1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True), )

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        layers = []
        layers.append(TRBlock(in_ch, out_ch, stride, shortcut))

        for i in range(1, block_num):
            layers.append(TRBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)
        maxpool0 = self.maxpool0(X_00)
        X_10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool1(X_10)
        X_20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool2(X_20)
        X_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool3(X_30)
        X_40 = self.conv40(maxpool3)

        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)

        h1_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(X_00))))
        h2_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(X_10))))
        h3_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(X_20))))
        X_31 = torch.cat((X_31,h1_hd4,h2_hd4,h3_hd4),1)

        X_31 = self.conv31(X_31)

        # column : 2
        X_02 = self.up_concat02(X_11, X_01)
        X_12 = self.up_concat12(X_21, X_11)
        X_22 = self.up_concat22(X_31, X_21)

        h1_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(X_00))))
        h2_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(X_10))))
        X_22 = torch.cat((X_22,h1_hd3,h2_hd3),1)
        X_22 = self.conv22(X_22)

        # column : 3
        X_03 = self.up_concat03(X_12, X_02)
        X_13 = self.up_concat13(X_22, X_12)
        h1_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(X_00))))

        X_13 = torch.cat((X_13, h1_hd2), 1)
        X_13 = self.conv13(X_13)
        # column : 4
        X_04 = self.up_concat04(X_13, X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4


        if self.is_ds:
            return F.leaky_relu(final,0.2)
        else:
            return F.leaky_relu(final_4,0.2)

# ---------------LK-GAN-----------------------------------
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.models_mae import MaskedAutoencoderViT
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
def check_img_data_range(img):
    if img.dtype == np.uint8:
        return 255
    else:
        return 1.0
def drop_path(x, drop_prob, training=False, scale_by_keep=True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor
class AverageCounter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
class Atten(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, dim):
        super(Atten, self).__init__()
        self.Adp=nn.AdaptiveAvgPool2d((1))
        self.conv=nn.Sequential(
            nn.Conv2d(dim,dim//4,1,1),
            nn.GELU(),
            nn.Conv2d(dim//4, dim, 1, 1),
            nn.Sigmoid()
        )
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        input=x
        #C*H*W
        x=x*self.conv(self.Adp(x))
        #1*H*W
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out=input+x*self.sigmoid(max_out)
        return out
def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, eps=1e-6, affine=True)
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_
class RepLKBlock(nn.Module):
    def __init__(self, in_channels, dw_channels, block_lk_size,drop_path=0.):
        super(RepLKBlock,self).__init__()
        self.pw1 = nn.Sequential(
            nn.Conv2d(in_channels, dw_channels,1,1,0,groups=1),
            nn.InstanceNorm2d(dw_channels),
            nn.GELU(),
        )
        self.pw2 = nn.Sequential(
            nn.Conv2d(dw_channels,in_channels,1,1,0,groups=1),
            nn.InstanceNorm2d(in_channels),
            nn.GELU(),
        )
        self.large_kernel = nn.Conv2d(in_channels=dw_channels, out_channels=dw_channels, kernel_size=block_lk_size,
                                                  stride=1,padding=block_lk_size // 2 ,groups=dw_channels,bias=True)
        self.lk_nonlinear = nn.GELU()
        self.prelkb_bn = nn.InstanceNorm2d(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #print('drop path:', self.drop_path)
    def forward(self, x):
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)
class ConvFFN(nn.Module):
    def __init__(self, in_channels, internal_channels, out_channels,drop_path=0.):
        super(ConvFFN,self).__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.preffn_bn = nn.InstanceNorm2d(in_channels)
        self.pw1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels,1,1,0,groups=1),
            nn.InstanceNorm2d(internal_channels),
        )
        self.pw2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels,1,1,0,groups=1),
            nn.InstanceNorm2d(out_channels),
        )
        self.nonlinear = nn.GELU()

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)
class Decoder(nn.Module):
    def __init__(self, in_planes=1, out_planes=16):
        super(Decoder, self).__init__()
        self.in_planes=in_planes
        self.out_planes = out_planes
        self.dense=nn.Sequential(
            nn.Conv2d(out_planes * 4, out_planes * 2, 3, 1, 1),
            RepLKBlock(out_planes * 2, out_planes * 2, 13),
            ConvFFN(out_planes * 2, out_planes * 8, out_planes * 2),
            RepLKBlock(out_planes * 2, out_planes * 2, 13),
            ConvFFN(out_planes * 2, out_planes * 8, out_planes * 2),

        )
        self.dense2=nn.Sequential(
            nn.Conv2d(out_planes * 2, out_planes , 3, 1, 1),
            RepLKBlock(out_planes , out_planes , 13),
            ConvFFN(out_planes , out_planes * 4, out_planes),
            RepLKBlock(out_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 4, out_planes),
            RepLKBlock(out_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 4, out_planes),
            RepLKBlock(out_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 4, out_planes),
            RepLKBlock(out_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 4, out_planes),
            RepLKBlock(out_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 4, out_planes),
        )
        self.dense3=nn.Sequential(
            RepLKBlock(out_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 2, out_planes),
            RepLKBlock(out_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 2, out_planes),
        )
        self.end=nn.Sequential(
            nn.Conv2d(out_planes,3,3,1,1),
        )
        self.start=nn.Sequential(
            nn.Conv2d(in_planes,out_planes,3,1,1),
            nn.InstanceNorm2d(out_planes),
            nn.GELU()
        )
        self.encoder = Encoder(in_planes, out_planes)
    def forward(self,nir):

        fu_en=self.encoder(nir)

        ee=self.dense(F.interpolate(fu_en[0], scale_factor=2, mode="nearest")) + fu_en[1]
        ee1 = self.dense2(F.interpolate(ee, scale_factor=2, mode="nearest")) + fu_en[2]
        ee2 = self.dense3(ee1)
        return ee2
class Decoder2(nn.Module):
    def __init__(self, in_planes=1, out_planes=16):
        super(Decoder2, self).__init__()
        self.in_planes=in_planes
        self.out_planes = out_planes
        self.d=nn.Sequential(
            nn.Conv2d(out_planes,out_planes*2,3,1,1),
            nn.InstanceNorm2d(out_planes*2 ),
            nn.GELU(),
        )
        self.up=nn.Sequential(

            nn.Conv2d(self.out_planes * 2, self.out_planes, kernel_size=3, stride=1,padding=1),
            nn.InstanceNorm2d(self.out_planes),
            nn.GELU(),
        )
        self.d2=nn.Sequential(
            nn.Conv2d(out_planes* 2,out_planes*4,3,2,1),
            nn.InstanceNorm2d(out_planes * 4),
            nn.GELU(),
        )
        self.up2=nn.Sequential(

            nn.Conv2d(self.out_planes * 4, self.out_planes*2, kernel_size=3, stride=1,padding=1),
            nn.InstanceNorm2d(self.out_planes*2),

            nn.GELU(),

        )
        self.d3=nn.Sequential(
            nn.Conv2d(out_planes* 4,out_planes*8,3,2,1),
            nn.InstanceNorm2d(out_planes * 8),
            nn.GELU(),
        )
        self.up3=nn.Sequential(
            nn.Conv2d(self.out_planes * 8, self.out_planes*4, kernel_size=3, stride=1,padding=1),
            nn.InstanceNorm2d(self.out_planes*4),
            nn.GELU(),

        )
        self.d4=nn.Sequential(
            nn.Conv2d(out_planes * 8, out_planes * 16, 3, 2, 1),
            nn.InstanceNorm2d(out_planes * 16),
            nn.GELU(),
        )
        self.up4=nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.out_planes * 16, self.out_planes*8, kernel_size=3, stride=1,padding=1),
            nn.InstanceNorm2d(self.out_planes*8),
            nn.GELU(),
        )
        self.out=nn.Sequential(
            nn.Conv2d(out_planes,3,1,1)
        )
        self.ct1=Atten(out_planes*2)
        self.ct2 = Atten(out_planes*4)
        self.ct3 = Atten(out_planes*8 )
    def forward(self,input):
        x1=self.ct1(self.d(input))
        x2=self.ct2(self.d2(x1))
        x3=self.ct3(self.d3(x2))
        x31=self.up3(F.interpolate(x3, scale_factor=2, mode="nearest"))+x2
        x21=self.up2(F.interpolate(x31, scale_factor=2, mode="nearest"))+x1
        x11=self.up(F.interpolate(x21, scale_factor=2, mode="nearest"))
        return x11
class Encoder(nn.Module):
    def __init__(self, in_planes=1, out_planes=16):
        super(Encoder, self).__init__()
        self.in_planes=in_planes
        self.out_planes = out_planes
        self.down1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
            nn.InstanceNorm2d(out_planes),
            nn.GELU(),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_planes),
            nn.GELU(),
        )
        self.encoder_layers1 = nn.Sequential(
            RepLKBlock(out_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 4, out_planes),
            RepLKBlock(out_planes, out_planes, 13),
            ConvFFN(out_planes, out_planes * 4, out_planes),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes * 2, kernel_size=1, stride=1),
            nn.InstanceNorm2d(out_planes * 2),
            nn.GELU(),
            nn.Conv2d(out_planes*2, out_planes * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_planes * 2),
            nn.GELU(),
        )
        self.encoder_layers2 = nn.Sequential(
            RepLKBlock(out_planes * 2, out_planes * 2, 13),
            ConvFFN(out_planes * 2, out_planes * 8, out_planes * 2),

            RepLKBlock(out_planes * 2, out_planes * 2, 13),
            ConvFFN(out_planes * 2, out_planes * 8, out_planes * 2),

            RepLKBlock(out_planes * 2, out_planes * 2, 13),
            ConvFFN(out_planes * 2, out_planes * 8, out_planes * 2),
            RepLKBlock(out_planes * 2, out_planes * 2, 13),
            ConvFFN(out_planes * 2, out_planes * 8, out_planes * 2),
            RepLKBlock(out_planes * 2, out_planes * 2, 13),
            ConvFFN(out_planes * 2, out_planes * 8, out_planes * 2),
            RepLKBlock(out_planes * 2, out_planes * 2, 13),
            ConvFFN(out_planes * 2, out_planes * 8, out_planes * 2),

        )
        self.down3 = nn.Sequential(
            nn.Conv2d(out_planes * 2, out_planes * 4, kernel_size=1, stride=1),
            nn.InstanceNorm2d(out_planes * 4),
            nn.GELU(),
            nn.Conv2d(out_planes * 4, out_planes * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_planes * 4),
            nn.GELU(),
        )
        self.encoder_layers3 = nn.Sequential(
            RepLKBlock(out_planes * 4, out_planes * 4, 13),
            ConvFFN(out_planes * 4, out_planes * 16, out_planes * 4),
            RepLKBlock(out_planes * 4, out_planes * 4, 13),
            ConvFFN(out_planes * 4, out_planes * 16, out_planes * 4),
        )
    def forward(self,input):
        e=self.down1(input)
        e=self.encoder_layers1(e)
        e11 = self.down2(e)
        e11=self.encoder_layers2(e11)
        e22 = self.down3(e11)
        e22=self.encoder_layers3(e22)
        return e22,e11,e


# Defines the Generator.
class LKAT_network(nn.Module):
    def __init__(self,in_planes=3,out_planes=64):
        super(LKAT_network, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.decoder1=Decoder(in_planes,out_planes)
        self.decoder2=Decoder2(out_planes,out_planes)
        self.Ence=MaskedAutoencoderViT(img_size=128,
        patch_size=16, in_chans=out_planes,embed_dim=512, depth=12, num_heads=8,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.out=nn.Conv2d(out_planes*2,3,1,1)
    def forward(self,nir_image):
        x2=nir_image

        mid=self.decoder1(x2)

        d2 = self.decoder2(mid)

        f_rgb_ence=self.Ence(mid)
        f=torch.cat((d2,f_rgb_ence),dim=1)
        f_rgb = self.out(f)
        return f_rgb

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator_LK(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator_LK, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
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
    def forward(self, input):
        return self.model(input)

# ------------------------------------------TICC-GAN----------------------------------------

# gloal local generator
class LocalEnhancer(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
	             n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
		super(LocalEnhancer, self).__init__()
		self.n_local_enhancers = n_local_enhancers

		###### global generator model #####
		ngf_global = ngf * (2 ** n_local_enhancers)
		model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
		                               norm_layer).model
		model_global = [model_global[i] for i in
		                range(len(model_global) - 3)]  # get rid of final convolution layers
		self.model = nn.Sequential(*model_global)

		###### local enhancer layers #####
		for n in range(1, n_local_enhancers + 1):
			### downsample
			ngf_global = ngf * (2 ** (n_local_enhancers - n))
			model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
			                    norm_layer(ngf_global), nn.ReLU(True),
			                    nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
			                    norm_layer(ngf_global * 2), nn.ReLU(True)]
			### residual blocks
			model_upsample = []
			for i in range(n_blocks_local):
				model_upsample += [ResnetBlock_gll(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

			### upsample
			model_upsample += [
				nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
				norm_layer(ngf_global), nn.ReLU(True)]

			### final convolution
			if n == n_local_enhancers:
				model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
				                   nn.Tanh()]

			setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))
			setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

		self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

	def forward(self, input):
		### create input pyramid
		input_downsampled = [input]
		for i in range(self.n_local_enhancers):
			input_downsampled.append(self.downsample(input_downsampled[-1]))

		### output at coarest level
		output_prev = self.model(input_downsampled[-1])
		### build up one layer at a time
		for n_local_enhancers in range(1, self.n_local_enhancers + 1):
			model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
			model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
			input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
			output_prev = model_upsample(model_downsample(input_i) + output_prev)
		return output_prev


class GlobalGenerator(nn.Module):   # 不用改名字
	def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
	             padding_type='reflect'):
		assert (n_blocks >= 0)
		super(GlobalGenerator, self).__init__()
		activation = nn.ReLU(True)

		model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
		### downsample
		for i in range(n_downsampling):
			mult = 2 ** i
			model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
			          norm_layer(ngf * mult * 2), activation]

		### resnet blocks
		mult = 2 ** n_downsampling
		for i in range(n_blocks):
			model += [
				ResnetBlock_gll(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

		### upsample
		for i in range(n_downsampling):
			mult = 2 ** (n_downsampling - i)
			model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
			                             output_padding=1),
			          norm_layer(int(ngf * mult / 2)), activation]
		model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
		self.model = nn.Sequential(*model)

	def forward(self, input):
		return self.model(input)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator_TICC(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
		super(NLayerDiscriminator_TICC, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		kw = 4
		padw = 1
		#先进行一次通道数不变，但是信息量进行了一次总结，图片尺寸发生变化，取决于卷积，4 2 1尺寸长宽缩小为原来的一倍
		sequence = [
			nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),#4*4*32
			nn.LeakyReLU(0.2, True)#（αx=0.2x,x）
		]

		nf_mult = 1
		nf_mult_prev = 1
		#三次卷积将通道数以2的倍数扩大到原来的8倍
		for n in range(1, n_layers):
			nf_mult_prev = nf_mult
			nf_mult = min(2 ** n, 8)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
				          kernel_size=kw, stride=2, padding=padw, bias=use_bias),#4*4*64-->4*4*128-->4*4*256
				norm_layer(ndf * nf_mult),
				nn.LeakyReLU(0.2, True)
			]

		nf_mult_prev = nf_mult#此时nf_mult为8
		nf_mult = min(2 ** n_layers, 8)
		#进行一次想同通道的卷积，将图片的尺寸进行改变，长宽-1
		sequence += [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
			          kernel_size=kw, stride=1, padding=padw, bias=use_bias),#4*4*256
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]
		#将最后的卷积进行一个通道的缩放，长宽-1
		sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]#4*4*1

		#是否使用Sigmoid激活函数，将结果强行压到0-1之间
		if use_sigmoid:
			sequence += [nn.Sigmoid()]

		self.model = nn.Sequential(*sequence)

	def forward(self, input):
		return self.model(input)



class cascaded(nn.Module):

	def __init__(self, input_nc, output_nc, ngf):
		super(cascaded, self).__init__()

		# Layer1 4*4---8*8
		self.conv1 = nn.Conv2d(input_nc, ngf * 16, kernel_size=3, stride=1, padding=1, bias=True)
		self.lay1 = LayerNorm(ngf * 16, eps=1e-12, affine=True)
		self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

		self.conv11 = nn.Conv2d(ngf * 16, ngf * 16, kernel_size=3, stride=1, padding=1, bias=True)
		self.lay11 = LayerNorm(ngf * 16, eps=1e-12, affine=True)
		self.relu11 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

		# Layer2 8*8---16*16
		self.conv2 = nn.Conv2d(ngf * 16 + input_nc, ngf * 16, kernel_size=3, stride=1, padding=1, bias=True)
		self.lay2 = LayerNorm(ngf * 16, eps=1e-12, affine=True)
		self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

		self.conv22 = nn.Conv2d(ngf * 16, ngf * 16, kernel_size=3, stride=1, padding=1, bias=True)
		self.lay22 = LayerNorm(ngf * 16, eps=1e-12, affine=True)
		self.relu22 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

		# layer3 16*16---32*32
		self.conv3 = nn.Conv2d(ngf * 16 + input_nc, ngf * 8, kernel_size=3, stride=1, padding=1, bias=True)
		self.lay3 = LayerNorm(ngf * 8, eps=1e-12, affine=True)
		self.relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

		self.conv33 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, bias=True)
		self.lay33 = LayerNorm(ngf * 8, eps=1e-12, affine=True)
		self.relu33 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

		# layer4 32*32---64*64
		self.conv4 = nn.Conv2d(ngf * 8 + input_nc, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True)
		self.lay4 = LayerNorm(ngf * 4, eps=1e-12, affine=True)
		self.relu4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

		self.conv44 = nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True)
		self.lay44 = LayerNorm(ngf * 4, eps=1e-12, affine=True)
		self.relu44 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

		# layer5 64*64---128*128

		self.conv5 = nn.Conv2d(ngf * 4 + input_nc, ngf * 2, kernel_size=3, stride=1, padding=1, bias=True)
		self.lay5 = LayerNorm(ngf * 2, eps=1e-12, affine=True)
		self.relu5 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

		self.conv55 = nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, bias=True)
		self.lay55 = LayerNorm(ngf * 2, eps=1e-12, affine=True)
		self.relu55 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

		# layer6 128*128---256*256
		self.conv6 = nn.Conv2d(ngf * 2 + input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=True)
		self.lay6 = LayerNorm(ngf, eps=1e-12, affine=True)
		self.relu6 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

		self.conv66 = nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=True)
		self.lay66 = LayerNorm(ngf, eps=1e-12, affine=True)
		self.relu66 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

		# Layer7 256*256
		self.conv7 = nn.Conv2d(ngf + input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=True)

		# Layer_downsample
		self.downsample = nn.AvgPool2d(kernel_size=2, padding=0, stride=2)

	def forward(self, input):
		input_128 = self.downsample(input)
		input_64 = self.downsample(input_128)
		input_32 = self.downsample(input_64)
		input_16 = self.downsample(input_32)
		input_8 = self.downsample(input_16)
		input_4 = self.downsample(input_8)

		# Layer1 4*4---8*8
		out1 = self.conv1(input_4)
		L1 = self.lay1(out1)
		out2 = self.relu1(L1)

		out11 = self.conv11(out2)
		L11 = self.lay11(out11)
		out22 = self.relu11(L11)

		m = nn.Upsample(size=(input_4.size(3) * 2, input_4.size(3) * 2), mode='bilinear')

		img1 = torch.cat((m(out22), input_8), 1)

		# Layer2 8*8---16*16
		out3 = self.conv2(img1)
		L2 = self.lay2(out3)
		out4 = self.relu2(L2)

		out33 = self.conv22(out4)
		L22 = self.lay22(out33)
		out44 = self.relu22(L22)

		m = nn.Upsample(size=(input_8.size(3) * 2, input_8.size(3) * 2), mode='bilinear')

		img2 = torch.cat((m(out44), input_16), 1)

		# Layer3 16*16---32*32
		out5 = self.conv3(img2)
		L3 = self.lay3(out5)
		out6 = self.relu3(L3)

		out55 = self.conv33(out6)
		L33 = self.lay33(out55)
		out66 = self.relu33(L33)

		m = nn.Upsample(size=(input_16.size(3) * 2, input_16.size(3) * 2), mode='bilinear')

		img3 = torch.cat((m(out66), input_32), 1)

		# Layer4 32*32---64*64
		out7 = self.conv4(img3)
		L4 = self.lay4(out7)
		out8 = self.relu4(L4)

		out77 = self.conv44(out8)
		L44 = self.lay44(out77)
		out88 = self.relu44(L44)

		m = nn.Upsample(size=(input_32.size(3) * 2, input_32.size(3) * 2), mode='bilinear')

		img4 = torch.cat((m(out88), input_64), 1)

		# Layer5 64*64---128*128
		out9 = self.conv5(img4)
		L5 = self.lay5(out9)
		out10 = self.relu5(L5)

		out99 = self.conv55(out10)
		L55 = self.lay55(out99)
		out110 = self.relu55(L55)

		m = nn.Upsample(size=(input_64.size(3) * 2, input_64.size(3) * 2), mode='bilinear')

		img5 = torch.cat((m(out110), input_128), 1)

		# Layer6 128*128---256*256
		out11 = self.conv6(img5)
		L6 = self.lay6(out11)
		out12 = self.relu6(L6)

		out111 = self.conv66(out12)
		L66 = self.lay66(out111)
		out112 = self.relu66(L66)

		m = nn.Upsample(size=(input_128.size(3) * 2, input_128.size(3) * 2), mode='bilinear')

		img6 = torch.cat((m(out112), input), 1)

		# Layer7 256*256
		out13 = self.conv7(img6)

		return out13
