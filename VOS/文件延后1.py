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
def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal',
             gpu_ids=[]):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'MSUNet':
        netG = MSUNet()
    elif which_model_netG == 'MSUNet_me':
        netG = MSUNet_me()
    elif which_model_netG == 'MSUNet_me_3':
        netG = MSUNet_me_3()
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'multi':
        netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D=3, getIntermFeat=False)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, gpu_ids)


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

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
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
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
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

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # print("NlayerDis")
        return self.model(input)


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


# 对参数进行选择创建两个conv1和conv2不同的卷积层
class unetConv2_2(nn.Module):          #  self.conv = unetConv2_2(in_size + (n_concat - 2) * out_size, out_size, False)
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2_2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:            # 创建两个卷积块分别为conv1和conv2
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.GELU(), )
                setattr(self, 'conv%d' % i, conv)   # 对上述创建的卷积进行该换名字
                in_size = out_size

        else:                         # 使用这个
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),        # 3，1，1 尺寸不变
                                     nn.GELU(), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')   #  对模块的所有子层(即卷积层),使用 Kaiming 初始化方法进行参数初始化。

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)       # 改名字
            if hasattr(torch.cuda,'empty_cache'):    # 检测CUDA缓存是否够用，然后再在每次进行卷积操作之后删除缓存
                torch.cuda.empty_cache()
            x = conv(x)
        return x

class unetConv2_com(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1,BN =True):
        super(unetConv2_com, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        self.BN = BN
        self.com = PAAttention(channel=out_size, reduction=8,kernel_size=7,BN =self.BN)

        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.GELU(), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:            # 使用这个
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.GELU(), )
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
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):         # 输入通道数，输出通道数
        super(unetUp_origin, self).__init__()
        if is_deconv:  # Ture
            self.conv = unetConv2_2(in_size + (n_concat - 2) * out_size, out_size, False)         # 两次卷积，尺寸不变
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)   # 两次反卷积，尺寸发生变化  （1，in_size,w，h）->（1，out_size，2w,2h）
        else:
            self.conv = unetConv2_2(in_size + (n_concat - 2) * out_size, out_size, False)         # 两次卷积，尺寸不变
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)                                     #与转置卷积相比,双线性插值上采样计算相对简单,但生成质量可能不如学习的转置卷积。实际应用中需要权衡计算开销和生成质量的tradeoff来选择合适的上采样方式。

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue    # 跳过名字包含unetConv2名称的模块
            init_weights(m, init_type='kaiming')                         # 对其他模块进行kaiming初始化

    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)                                      # 进行上采样
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)

class unetUp_origin_end(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2,BN =True):
        super(unetUp_origin_end, self).__init__()
        self.BN = BN
        if is_deconv:                    # True
            self.conv = unetConv2_com(in_size + (n_concat - 2) * out_size, out_size, False,BN=self.BN)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConv2_com(in_size + (n_concat - 2) * out_size, out_size, False,BN=self.BN)
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


# 文章提出的下采样模块
# 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
class TRBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(TRBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
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
        self.dila = nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,padding=1,dilation=1)  # 因为之后没有接BN所以需要使用偏执
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
        residual = x if self.right is None else self.right(x)  # 这就是残差块的核心思想:通过引入捷径分支来保存输入的"残差"信息,并将其与主分支的输出相加,从而能够更好地训练和优化深层神经网络。
        out += residual
        return F.relu(out)     # import torch.nn.functional as F


class MSUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=3, feature_scale=4, is_deconv=True, is_batchnorm=True, is_ds=True,BN=False):
        super(MSUNet, self).__init__()
        self.is_deconv = is_deconv                # True
        self.in_channels = in_channels            # 3
        self.is_batchnorm = is_batchnorm          # True
        self.is_ds = is_ds                        # True
        self.feature_scale = feature_scale        # 4
        self.CatChannels = 64
        self.BN = BN
        filters = [32, 64, 128, 256, 512]

        # downsampling
        self.conv00 = self.make_layer(self.in_channels, filters[0], 1)     # （3，32，2） 2代表使用2个TRB  （1，3，w，h）  ->  （1，32，w，h）
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)                        #  池化核为2的最大池化          （1，32，w，h）  ->  （1，32，w/2，h/2）
        self.conv10 = self.make_layer(filters[0], filters[1], 1)           #                            （1，32，w/2，h/2）  ->  （1，64，w/2，h/2）
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)                        #  池化核为2的最大池化          （1，64，w/2，h/2）  ->  （1，64，w/4，h/4）
        self.conv20 = self.make_layer(filters[1], filters[2], 1)           #                            （1，64，w/4，h/4）  ->  （1，128，w/4，h/4）
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)                        #  池化核为2的最大池化          （1，128，w/4，h/4）  ->  （1，128，w/8，h/8）
        self.conv30 = self.make_layer(filters[2], filters[3], 1)           #                            （1，128，w/8，h/8）  ->  （1，256，w/8，h/8）
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)                        #  池化核为2的最大池化          （1，256，w/8，h/8）  ->  （1，256，w/16，h/16）
        self.conv40 = self.make_layer(filters[3], filters[4], 1)           #                            （1，256，w/16，h/16）  ->  （1，512，w/16，h/16）

        # upsampling  单层次上采样
        self.up_concat01 = unetUp_origin(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp_origin(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp_origin(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp_origin_end(filters[4], filters[3], self.is_deconv,BN=self.BN)         # 添加注意力模块

        self.up_concat02 = unetUp_origin(filters[1], filters[0], self.is_deconv, 2)
        self.up_concat12 = unetUp_origin(filters[2], filters[1], self.is_deconv, 2)
        self.up_concat22 = unetUp_origin_end(filters[3], filters[2], self.is_deconv, 2,BN=self.BN)       # 添加注意力模块

        self.up_concat03 = unetUp_origin(filters[1], filters[0], self.is_deconv, 2)
        self.up_concat13  = unetUp_origin_end(filters[2], filters[1], self.is_deconv, 2,BN=self.BN)      # 添加注意力模块

        self.up_concat04 = unetUp_origin_end(filters[1], filters[0], self.is_deconv, 2,BN=self.BN)       # 添加注意力模块

        #upsampling_add
        #layer4
        # 第一行的向第二行进行最大池化链接  需要三个
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)                                     # ceil_mode=True 表示输出特征图的大小将向上取整 (1,a,w,h)->(1,a,w/8,h/8) 并且向下取整
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)             # 卷积 （1，32，w，h）  ->  （1，64，w，h）
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)                                    # 归一化 64个通道
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)                                             # 激活层


        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)                                     # ceil_mode=True 表示输出特征图的大小将向上取整 (1,a,w,h)->(1,a,w/4,h/4) 并且向下取整
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)             # 卷积 （1，64，w，h）  ->  （1，64，w，h）
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)                                    # 归一化 64个通道
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)                                             # 激活层

        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)                                     # ceil_mode=True 表示输出特征图的大小将向上取整 (1,a,w,h)->(1,a,w/2,h/2) 并且向下取整
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)             # 卷积 （1，128，w，h）  ->  （1，64，w，h）
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)                                    # 归一化 64个通道
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)                                             # 激活层

        # self.conv31 = nn.Conv2d(448, 256, 1)
        self.conv31 = nn.Sequential(nn.Conv2d(448, 256, 1),                                     # 改变通道数
                             nn.BatchNorm2d(256),
                             nn.ReLU(inplace=True), )

        # layer3
        # 第二行的向第三行进行最大池化链接  需要2个
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
        # 第三行的向第四行进行最大池化链接  需要2个
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # self.conv13 = nn.Conv2d(128, 64, 1)
        self.conv13 = nn.Sequential(nn.Conv2d(128, 64, 1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True), )

        # final conv (without any concat)  将通道数降为输出的三层,最上面四个输出
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights  初始化网络
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
    #
    # def make_layer(self, in_ch, out_ch, block_num, stride=1):    # 完成一个残差块，包括 2个TRB的残差连接 ，但是输出图像的尺寸没有发生变化
    #     shortcut = nn.Sequential(
    #         nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 相当于一个残差连接
    #         nn.BatchNorm2d(out_ch)
    #     )
    #     layers = []
    #     layers.append(TRBlock(in_ch, out_ch, stride, shortcut))
    #
    #     for i in range(1, block_num):
    #         layers.append(TRBlock(out_ch, out_ch))
    #     return nn.Sequential(*layers)

    def make_layer(self, in_ch, out_ch, block_num, stride=1,BN = True):    # 完成一个残差块，包括 2个TRB的残差连接 ，但是输出图像的尺寸没有发生变化
        shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 相当于一个残差连接
            nn.BatchNorm2d(out_ch)
        )
        self.BN = BN
        layers = []
        layers.append(TRBlock(in_ch, out_ch, stride, shortcut))

        for i in range(1, block_num):
            layers.append(TRBlock(in_ch, out_ch, stride, shortcut))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)                            # （1，3，w，h）  ->  （1，32，w，h）
        maxpool0 = self.maxpool0(X_00)                        # （1，32，w，h）  ->  （1，32，w/2，h/2）
        X_10 = self.conv10(maxpool0)                          # （1，32，w/2，h/2）  ->  （1，64，w/2，h/2）
        maxpool1 = self.maxpool1(X_10)                        # （1，64，w/2，h/2）  ->  （1，64，w/4，h/4）
        X_20 = self.conv20(maxpool1)                          # （1，64，w/4，h/4）  ->  （1，128，w/4，h/4）
        maxpool2 = self.maxpool2(X_20)                        # （1，128，w/4，h/4）  ->  （1，128，w/8，h/8）
        X_30 = self.conv30(maxpool2)                          # （1，128，w/8，h/8）  ->  （1，256，w/8，h/8）
        maxpool3 = self.maxpool3(X_30)                        # （1，256，w/8，h/8）  ->  （1，256，w/16，h/16）
        X_40 = self.conv40(maxpool3)                          # （1，256，w/16，h/16）  ->  （1，512，w/16，h/16）

        # column : 1
        X_01 = self.up_concat01(X_10, X_00)                   # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，32，w，h）
        X_11 = self.up_concat11(X_20, X_10)                   # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，64，w/2，h/2）
        X_21 = self.up_concat21(X_30, X_20)                   # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，128，w/4，h/4）
        X_31 = self.up_concat31(X_40, X_30)                   # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，256，w/8，h/8）

        h1_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(X_00))))
        h2_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(X_10))))
        h3_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(X_20))))
        X_31 = torch.cat((X_31,h1_hd4,h2_hd4,h3_hd4),1)             # 256+64+64+64= 448

        X_31 = self.conv31(X_31)                                    # （1，448，w/8，h/8）  ->  （1，256，w/8，h/8）

        # column : 2
        X_02 = self.up_concat02(X_11, X_01)                   # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，32，w，h）
        X_12 = self.up_concat12(X_21, X_11)                   # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，64，w/2，h/2）
        X_22 = self.up_concat22(X_31, X_21)                   # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，128，w/4，h/4）

        h1_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(X_00))))
        h2_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(X_10))))
        X_22 = torch.cat((X_22,h1_hd3,h2_hd3),1)              # 128+64+64 =256
        X_22 = self.conv22(X_22)                              # （1，256，w/4，h/4）  ->  （1，128，w/4，h/4）

        # column : 3
        X_03 = self.up_concat03(X_12, X_02)                   # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，32，w，h）
        X_13 = self.up_concat13(X_22, X_12)                   # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，64，w/2，h/2）
        h1_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(X_00))))

        X_13 = torch.cat((X_13, h1_hd2), 1)                  # 64+64 = 128
        X_13 = self.conv13(X_13)                             # （1，128，w/2，h/2）  ->  （1，64，w/2，h/2）
        # column : 4
        X_04 = self.up_concat04(X_13, X_03)                  # (1,32，w，h)

        # final layer  将最上面的四个输出进行3通道的变换
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4    # 将取得的彩色化结果进行最上面的4个输出进行平均叠加

        # image_numpy = util.tensor2im(final)
        #
        # print('image_numpy','\n',image_numpy)
        #
        # cv2.imshow('Image', image_numpy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if self.is_ds:
            return F.leaky_relu(final,0.2)                         # 适当的保存负值避免梯度消失
        else:
            return F.leaky_relu(final_4,0.2)

# 定义一个下采样细节块
class DOWNlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super(DOWNlock, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )
        self.upmid = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=2, bias=False,dilation=2),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )
        self.downmid = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=2, bias=False,dilation=2),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 相当于一个残差连接
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )
        self.upmid2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=2, bias=False, dilation=2),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )
        self.downmid2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=2, bias=False, dilation=2),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

        self.dila0 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,padding=1,dilation=1), # 因为之后没有接BN所以需要使用偏执
            nn.BatchNorm2d(out_ch),
        )
        self.dila = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1, dilation=1)
        self.final = nn.Conv2d(in_channels=7*out_ch,out_channels=out_ch,kernel_size=1,stride=1)


    def forward(self, x):

        shortcut_x = self.shortcut(x)

        # 上面第一行
        up1 = self.up(x)+ shortcut_x
        up2 = self.upmid(up1)
        up3 = self.upmid(up2) +up1
        # 下面第一行
        down1 = self.down(x) + shortcut_x
        down2 = self.downmid(down1)
        down3 = self.downmid(down2) +down1
        # 上面第二行
        up1_2 = self.up2(x) + shortcut_x
        up2_2 = self.upmid2(up1_2) + up1_2
        up3_2 = self.upmid2(up2_2) + up2_2
        # 下面第二行
        down1_2 = self.down2(x) + shortcut_x
        down2_2 = self.downmid2(down1_2) + down1_2
        down3_2 = self.downmid2(down2_2) + down2_2

        # 中间
        mid1 = up1 * down1
        mid1 = self.dila0(mid1)

        mid2 = up2 * down2 * mid1
        mid2 = self.dila(mid2)
        # 中间上
        mid3 = up2_2 * up2
        mid3 = self.dila(mid3)
        # 中间下
        mid4 = down2 * down2_2
        mid4 = self.dila(mid4)


        out = torch.cat((up3, down3, mid2,up3_2,down3_2,mid3,mid4), 1)

        out = self.final(out)
        out = out + shortcut_x  # 残差连接

        return F.relu(out)     # import torch.nn.functional as F

class guolvkuai(nn.Module):  # 定义过滤块

    def __init__(self, in_ch, out_ch , t_4,to_4 = False):
        super(guolvkuai, self).__init__()

        self.t_4 = t_4
        self.to_4 = to_4

        # 通道由 N->1 由于输入的数据已经经过 BN-R，因此不用加BN-LR
        self.N_1 = nn.Sequential(
            nn.Conv2d(in_ch, 1, 1, stride=1, padding=0, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        # 通道由 1->N 此时加BN-LR
        self._1_N_2 = nn.Sequential(
            nn.Conv2d(1, out_ch, 1, stride=1, padding=0, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.N_N_3_3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.N_N_5_5 = nn.Sequential(
            nn.Conv2d(1, 1, 5, stride=1, padding=2, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        self.N_N_7_7 = nn.Sequential(
            nn.Conv2d(1, 1, 7, stride=1, padding=3, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        self.I_N_4 = nn.Sequential(
            nn.Conv2d(out_ch, self.t_4, 1, stride=1, padding=0, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
            nn.BatchNorm2d(self.t_4),
            nn.ReLU(inplace=True),
        )
    def forward(self, inputs):

        if self.to_4:
            x = self.I_N_4(inputs)
        else:
            x_3 = self.N_N_3_3(inputs)
            x_5 = self._1_N_2(self.N_N_5_5(self.N_1(inputs)))
            x_7 = self._1_N_2(self.N_N_7_7(self.N_1(inputs)))

            x = x_3 + x_5 + x_7

        return x
class Canchakuai(nn.Module):

    def __init__(self, in_ch, out_ch, stride = 1,shortcut=None):
        super(Canchakuai, self).__init__()

        self.right = shortcut
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.out = [out_ch,int(out_ch/2),int(out_ch/4),int(out_ch/8),int(out_ch/16)]


       # 最开始的1×1的卷积块 此时通道输出为 M->N
        self.M_N = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.guolukuai_1 = guolvkuai(self.out[0], self.out[1],self.out[2],to_4=False)
        self.guolukuai_2 = guolvkuai(self.out[1], self.out[2],self.out[2],to_4=False)
        self.guolukuai_3 = guolvkuai(self.out[2], self.out[3],self.out[2],to_4=False)
        self.guolukuai_4 = guolvkuai(self.out[3], self.out[4],self.out[2],to_4=False)

        self.guolukuai_1_4 = guolvkuai(self.out[0], self.out[1], self.out[2], to_4=True)
        self.guolukuai_2_4 = guolvkuai(self.out[1], self.out[2], self.out[2], to_4=True)
        self.guolukuai_3_4 = guolvkuai(self.out[2], self.out[3], self.out[2], to_4=True)
        self.guolukuai_4_4 = guolvkuai(self.out[3], self.out[4], self.out[2], to_4=True)



  # self.out = [out_ch, int(out_ch / 2), int(out_ch / 4), int(out_ch / 8),int(out_ch / 16)]
    def forward(self, x):

        x_M_N = self.M_N(x)
        guolukuai1 = self.guolukuai_1(x_M_N)
        guolukuai2 = self.guolukuai_2(guolukuai1)
        guolukuai3 = self.guolukuai_3(guolukuai2)
        guolukuai4 = self.guolukuai_4(guolukuai3)

        guolukuai1_4 = self.guolukuai_1_4(guolukuai1)
        guolukuai2_4 = self.guolukuai_2_4(guolukuai2)
        guolukuai3_4 = self.guolukuai_3_4(guolukuai3)
        guolukuai4_4 = self.guolukuai_4_4(guolukuai4)

        x = torch.cat((guolukuai1_4, guolukuai2_4,guolukuai3_4,guolukuai4_4), 1)

        y = x + x_M_N

        return F.relu(y)



class MSUNet_me(nn.Module):
    def __init__(self, in_channels=3, n_classes=3, feature_scale=4, is_deconv=True, is_batchnorm=True, is_ds=True,BN = False):
        super(MSUNet_me, self).__init__()
        self.is_deconv = is_deconv                # True
        self.in_channels = in_channels            # 3
        self.is_batchnorm = is_batchnorm          # True
        self.is_ds = is_ds                        # True
        self.feature_scale = feature_scale        # 4
        self.CatChannels = 64
        self.BN = BN
        filters = [32, 64, 128, 256, 512]

        # downsampling
        self.conv00 = self.make_layer(self.in_channels, filters[0], 3,BN=self.BN)     # （3，32，2） 2代表使用2个TRB  （1，3，w，h）  ->  （1，32，w，h）
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)                        #  池化核为2的最大池化          （1，32，w，h）  ->  （1，32，w/2，h/2）
        self.conv10 = self.make_layer(filters[0], filters[1], 3,BN=self.BN)           #                            （1，32，w/2，h/2）  ->  （1，64，w/2，h/2）
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)                        #  池化核为2的最大池化          （1，64，w/2，h/2）  ->  （1，64，w/4，h/4）
        self.conv20 = self.make_layer(filters[1], filters[2], 3,BN=self.BN)           #                            （1，64，w/4，h/4）  ->  （1，128，w/4，h/4）
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)                        #  池化核为2的最大池化          （1，128，w/4，h/4）  ->  （1，128，w/8，h/8）
        self.conv30 = self.make_layer(filters[2], filters[3], 3,BN=self.BN)           #                            （1，128，w/8，h/8）  ->  （1，256，w/8，h/8）
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)                        #  池化核为2的最大池化          （1，256，w/8，h/8）  ->  （1，256，w/16，h/16）
        self.conv40 = self.make_layer(filters[3], filters[4], 3,BN=self.BN)           #                            （1，256，w/16，h/16）  ->  （1，512，w/16，h/16）

        # upsampling  单层次上采样
        self.up_concat01 = unetUp_origin(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp_origin(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp_origin(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp_origin_end(filters[4], filters[3], self.is_deconv,BN=self.BN)         # 添加注意力模块

        self.up_concat02 = unetUp_origin(filters[1], filters[0], self.is_deconv, 2)
        self.up_concat12 = unetUp_origin(filters[2], filters[1], self.is_deconv, 2)
        self.up_concat22 = unetUp_origin_end(filters[3], filters[2], self.is_deconv, 2,BN=self.BN)       # 添加注意力模块

        self.up_concat03 = unetUp_origin(filters[1], filters[0], self.is_deconv, 2)
        self.up_concat13  = unetUp_origin_end(filters[2], filters[1], self.is_deconv, 2,BN=self.BN)      # 添加注意力模块

        self.up_concat04 = unetUp_origin_end(filters[1], filters[0], self.is_deconv, 2,BN=self.BN)       # 添加注意力模块

        #upsampling_add
        #layer4
        # 第一行的向第二行进行最大池化链接  需要三个
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)                                     # ceil_mode=True 表示输出特征图的大小将向上取整 (1,a,w,h)->(1,a,w/8,h/8) 并且向下取整
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)             # 卷积 （1，32，w，h）  ->  （1，64，w，h）
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)                                    # 归一化 64个通道
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)                                             # 激活层


        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)                                     # ceil_mode=True 表示输出特征图的大小将向上取整 (1,a,w,h)->(1,a,w/4,h/4) 并且向下取整
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)             # 卷积 （1，64，w，h）  ->  （1，64，w，h）
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)                                    # 归一化 64个通道
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)                                             # 激活层

        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)                                     # ceil_mode=True 表示输出特征图的大小将向上取整 (1,a,w,h)->(1,a,w/2,h/2) 并且向下取整
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)             # 卷积 （1，128，w，h）  ->  （1，64，w，h）
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)                                    # 归一化 64个通道
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)                                             # 激活层

        # self.conv31 = nn.Conv2d(448, 256, 1)
        self.conv31 = nn.Sequential(nn.Conv2d(448, 256, 1),                                     # 改变通道数
                             nn.BatchNorm2d(256),
                             nn.ReLU(inplace=True), )

        # layer3
        # 第二行的向第三行进行最大池化链接  需要2个
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
        # 第三行的向第四行进行最大池化链接  需要2个
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # self.conv13 = nn.Conv2d(128, 64, 1)
        self.conv13 = nn.Sequential(nn.Conv2d(128, 64, 1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True), )

        # final conv (without any concat)  将通道数降为输出的三层,最上面四个输出
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights  初始化网络
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
    #
    # def make_layer(self, in_ch, out_ch, block_num, stride=1):    # 完成一个残差块，包括 2个TRB的残差连接 ，但是输出图像的尺寸没有发生变化
    #     shortcut = nn.Sequential(
    #         nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 相当于一个残差连接
    #         nn.BatchNorm2d(out_ch)
    #     )
    #     layers = []
    #     layers.append(TRBlock(in_ch, out_ch, stride, shortcut))
    #
    #     for i in range(1, block_num):
    #         layers.append(TRBlock(out_ch, out_ch))
    #     return nn.Sequential(*layers)

    def make_layer(self, in_ch, out_ch, block_num, stride=1,BN = True):    # 完成一个残差块，包括 2个TRB的残差连接 ，但是输出图像的尺寸没有发生变化
        shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 相当于一个总体的残差连接
            nn.BatchNorm2d(out_ch)
        )
        self.BN = BN
        layers = []
        layers.append(Canchakuai_me(in_ch, out_ch,BN = self.BN,shortcut= shortcut))

        for i in range(1, block_num):
            layers.append(Canchakuai_me(out_ch, out_ch,BN = self.BN,shortcut= shortcut))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)                            # （1，3，w，h）  ->  （1，32，w，h）
        maxpool0 = self.maxpool0(X_00)                        # （1，32，w，h）  ->  （1，32，w/2，h/2）
        X_10 = self.conv10(maxpool0)                          # （1，32，w/2，h/2）  ->  （1，64，w/2，h/2）
        maxpool1 = self.maxpool1(X_10)                        # （1，64，w/2，h/2）  ->  （1，64，w/4，h/4）
        X_20 = self.conv20(maxpool1)                          # （1，64，w/4，h/4）  ->  （1，128，w/4，h/4）
        maxpool2 = self.maxpool2(X_20)                        # （1，128，w/4，h/4）  ->  （1，128，w/8，h/8）
        X_30 = self.conv30(maxpool2)                          # （1，128，w/8，h/8）  ->  （1，256，w/8，h/8）
        maxpool3 = self.maxpool3(X_30)                        # （1，256，w/8，h/8）  ->  （1，256，w/16，h/16）
        X_40 = self.conv40(maxpool3)                          # （1，256，w/16，h/16）  ->  （1，512，w/16，h/16）

        # column : 1
        X_01 = self.up_concat01(X_10, X_00)                   # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，32，w，h）
        X_11 = self.up_concat11(X_20, X_10)                   # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，64，w/2，h/2）
        X_21 = self.up_concat21(X_30, X_20)                   # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，128，w/4，h/4）
        X_31 = self.up_concat31(X_40, X_30)                   # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，256，w/8，h/8）

        h1_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(X_00))))
        h2_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(X_10))))
        h3_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(X_20))))
        X_31 = torch.cat((X_31,h1_hd4,h2_hd4,h3_hd4),1)             # 256+64+64+64= 448

        X_31 = self.conv31(X_31)                                    # （1，448，w/8，h/8）  ->  （1，256，w/8，h/8）

        # column : 2
        X_02 = self.up_concat02(X_11, X_01)                   # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，32，w，h）
        X_12 = self.up_concat12(X_21, X_11)                   # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，64，w/2，h/2）
        X_22 = self.up_concat22(X_31, X_21)                   # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，128，w/4，h/4）

        h1_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(X_00))))
        h2_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(X_10))))
        X_22 = torch.cat((X_22,h1_hd3,h2_hd3),1)              # 128+64+64 =256
        X_22 = self.conv22(X_22)                              # （1，256，w/4，h/4）  ->  （1，128，w/4，h/4）

        # column : 3
        X_03 = self.up_concat03(X_12, X_02)                   # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，32，w，h）
        X_13 = self.up_concat13(X_22, X_12)                   # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，64，w/2，h/2）
        h1_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(X_00))))

        X_13 = torch.cat((X_13, h1_hd2), 1)                  # 64+64 = 128
        X_13 = self.conv13(X_13)                             # （1，128，w/2，h/2）  ->  （1，64，w/2，h/2）
        # column : 4
        X_04 = self.up_concat04(X_13, X_03)                  # (1,32，w，h)

        # final layer  将最上面的四个输出进行3通道的变换
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4    # 将取得的彩色化结果进行最上面的4个输出进行平均叠加

        # image_numpy = util.tensor2im(final)
        #
        # print('image_numpy','\n',image_numpy)
        #
        # cv2.imshow('Image', image_numpy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if self.is_ds:
            return F.leaky_relu(final,0.2)                         # 适当的保存负值避免梯度消失
        else:
            return F.leaky_relu(final_4,0.2)


class guolvkuai_me(nn.Module):  # 定义过滤块

    def __init__(self, in_ch, out_ch , t_4,to_4 = False,BN = True):
        super(guolvkuai_me, self).__init__()

        self.t_4 = t_4
        self.to_4 = to_4

        if BN:
            # 通道由 N->1 由于输入的数据已经经过 BN-R，因此不用加BN-LR
            self.N_1 = nn.Sequential(
                nn.Conv2d(in_ch, 1, 1, stride=1, padding=0, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                nn.BatchNorm2d(1),
                nn.GELU(),
            )
            # 通道由 1->N 此时加BN-LR
            self._1_N_2 = nn.Sequential(
                nn.Conv2d(1, out_ch, 1, stride=1, padding=0, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            )

            self.N_N_3_3 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            )

            self.N_N_5_5 = nn.Sequential(
                nn.Conv2d(1, 1, 5, stride=1, padding=2, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                nn.BatchNorm2d(1),
                nn.GELU(),
            )

            self.N_N_7_7 = nn.Sequential(
                nn.Conv2d(1, 1, 7, stride=1, padding=3, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                nn.BatchNorm2d(1),
                nn.GELU(),
            )

            self.I_N_4 = nn.Sequential(
                nn.Conv2d(out_ch, self.t_4, 1, stride=1, padding=0, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                nn.BatchNorm2d(self.t_4),
                nn.GELU(),
            )
        else:
            # 通道由 N->1 由于输入的数据已经经过 BN-R，因此不用加BN-LR
            self.N_1 = nn.Sequential(
                nn.Conv2d(in_ch, 1, 1, stride=1, padding=0, bias=True),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                nn.GELU(),
            )
            # 通道由 1->N 此时加BN-LR
            self._1_N_2 = nn.Sequential(
                nn.Conv2d(1, out_ch, 1, stride=1, padding=0, bias=True),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                nn.GELU(),
            )

            self.N_N_3_3 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=True),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                nn.GELU(),
            )

            self.N_N_5_5 = nn.Sequential(
                nn.Conv2d(1, 1, 5, stride=1, padding=2, bias=True),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                nn.GELU(),
            )

            self.N_N_7_7 = nn.Sequential(
                nn.Conv2d(1, 1, 7, stride=1, padding=3, bias=True),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                nn.GELU(),
            )

            self.I_N_4 = nn.Sequential(
                nn.Conv2d(out_ch, self.t_4, 1, stride=1, padding=0, bias=True),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                nn.GELU(),
            )

    def forward(self, inputs):

        if self.to_4:
            x = self.I_N_4(inputs)
        else:
            x_3 = self.N_N_3_3(inputs)
            x_5 = self._1_N_2(self.N_N_5_5(self.N_1(inputs)))
            x_7 = self._1_N_2(self.N_N_7_7(self.N_1(inputs)))

            x = x_3 + x_5 + x_7

        return x
# class Canchakuai_me(nn.Module):
#
#     def __init__(self, in_ch, out_ch, BN = True,shortcut=None):
#         super(Canchakuai_me, self).__init__()
#
#         self.right = shortcut
#         self.in_ch = in_ch
#         self.out_ch = out_ch
#         # 假如输入为128通道 128 64 32 16 8
#         self.out = [out_ch,out_ch//2,out_ch//4,out_ch//8, out_ch//16]
#         self.BN = BN
#
#         if self.BN:
#            # 最开始的1×1的卷积块 此时通道输出为 M->N 即in_ch->out_ch
#             self.M_N = nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
#                 nn.BatchNorm2d(out_ch),
#                 nn.GELU(),
#             )
#             self.one_one = nn.Sequential(
#                 nn.Conv2d(out_ch, 1, 1, stride=1, padding=0, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
#                 nn.BatchNorm2d(1),
#                 nn.GELU(),
#                 nn.Conv2d(1, out_ch, 1, stride=1, padding=0, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
#                 nn.BatchNorm2d(out_ch),
#                 nn.GELU(),
#             )
#         else:
#             self.M_N = nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=True),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
#                 #nn.BatchNorm2d(out_ch),
#                 nn.GELU(),
#             )
#             self.one_one = nn.Sequential(
#                 nn.Conv2d(out_ch, 1, 1, stride=1, padding=0, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
#                 #nn.BatchNorm2d(1),
#                 nn.GELU(),
#                 nn.Conv2d(1, out_ch, 1, stride=1, padding=0, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
#                 #nn.BatchNorm2d(out_ch),
#                 nn.GELU(),
#             )
#         # 分别得到四个层不同通道的卷积结果
#         self.guolukuai_1 = guolvkuai_me(self.out[0], self.out[1],self.out[2],to_4=False,BN=True)
#         self.guolukuai_2 = guolvkuai_me(self.out[1], self.out[2],self.out[2],to_4=False,BN=True)
#         self.guolukuai_3 = guolvkuai_me(self.out[2], self.out[3],self.out[2],to_4=False,BN=True)
#         self.guolukuai_4 = guolvkuai_me(self.out[3], self.out[4],self.out[2],to_4=False,BN=True)
#         # 分别得到四个层相同通道的卷积结果，最后的结果为通道数为self.out[2]
#         self.guolukuai_1_4 = guolvkuai_me(self.out[0], self.out[1], self.out[2], to_4=True,BN=True)
#         self.guolukuai_2_4 = guolvkuai_me(self.out[1], self.out[2], self.out[2], to_4=True,BN=True)
#         self.guolukuai_3_4 = guolvkuai_me(self.out[2], self.out[3], self.out[2], to_4=True,BN=True)
#         self.guolukuai_4_4 = guolvkuai_me(self.out[3], self.out[4], self.out[2], to_4=True,BN=True)
#
#     def forward(self, x):
#
#         x_M_N = self.M_N(x)
#         guolukuai1 = self.guolukuai_1(x_M_N)
#         guolukuai2 = self.guolukuai_2(guolukuai1)
#         guolukuai3 = self.guolukuai_3(guolukuai2)
#         guolukuai4 = self.guolukuai_4(guolukuai3)
#
#         guolukuai1_4 = self.guolukuai_1_4(guolukuai1)
#         guolukuai2_4 = self.guolukuai_2_4(guolukuai2)
#         guolukuai3_4 = self.guolukuai_3_4(guolukuai3)
#         guolukuai4_4 = self.guolukuai_4_4(guolukuai4)
#
#         x = torch.cat((guolukuai1_4, guolukuai2_4,guolukuai3_4,guolukuai4_4), 1)
#
#         y = self.one_one(x)
#
#         return F.gelu(y)


class Canchakuai_me(nn.Module):

    def __init__(self, in_ch, out_ch, BN = True,shortcut=None, number = 1): # number为中间需要的降维数
        super(Canchakuai_me, self).__init__()

        self.right = shortcut
        self.in_ch = in_ch
        self.out_ch = out_ch
        # 假如输入为128通道 128 64 32 16 8
        self.BN = BN
        self.number = 1

        if self.BN:
           # 最开始的1×1的卷积块 此时通道输出为 M->N 即in_ch->out_ch,用于实现跳跃连接
            self.M_N = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            )
           # 用于实现将最后得到的结果进行整合
            self.one_one = nn.Sequential(
                nn.Conv2d(self.number * 4, self.number, 1, stride=1, padding=0, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                nn.BatchNorm2d(self.number),
                nn.Conv2d(self.number, out_ch, 1, stride=1, padding=0, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            )
           # 通道由 N->self.number 由于输入的数据已经经过 BN-R，因此不用加BN-LR
            self.N_1 = nn.Sequential(
                nn.Conv2d(in_ch, self.number, 1, stride=1, padding=0, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                nn.BatchNorm2d(self.number),
                nn.GELU(),
            )
            # 通道由 1->N 此时加BN-LR
            self._1_N_2 = nn.Sequential(
                nn.Conv2d(self.number, out_ch, 1, stride=1, padding=0, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            )
            # 使用三成三的卷积进行特征提取
            self.N_N_3_3 = nn.Sequential(
                nn.Conv2d(self.number, self.number, 3, stride=1, padding=1, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                nn.BatchNorm2d(self.number),
                nn.GELU(),
            )

        else:
            # 最开始的1×1的卷积块 此时通道输出为 M->N 即in_ch->out_ch,用于实现跳跃连接
            self.M_N = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=False),  # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                #nn.BatchNorm2d(out_ch),
                nn.GELU(),
            )
            # 用于实现将最后得到的结果进行整合
            self.one_one = nn.Sequential(
                nn.Conv2d(self.number * 4, self.number, 1, stride=1, padding=0, bias=False),
                # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                #nn.BatchNorm2d(self.number),
                nn.GELU(),
                nn.Conv2d(self.number, out_ch, 1, stride=1, padding=0, bias=False),
                # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                #nn.BatchNorm2d(out_ch),
                nn.GELU(),
            )
            # 通道由 N->self.number 由于输入的数据已经经过 BN-R，因此不用加BN-LR
            self.N_1 = nn.Sequential(
                nn.Conv2d(in_ch, self.number, 1, stride=1, padding=0, bias=False),
                # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                #nn.BatchNorm2d(self.number),
                nn.GELU(),
            )
            # 通道由 1->N 此时加BN-LR
            self._1_N_2 = nn.Sequential(
                nn.Conv2d(self.number, out_ch, 1, stride=1, padding=0, bias=False),
                # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                #nn.BatchNorm2d(out_ch),
                nn.GELU(),
            )
            # 使用三成三的卷积进行特征提取
            self.N_N_3_3 = nn.Sequential(
                nn.Conv2d(self.number, self.number, 3, stride=1, padding=1, bias=False),
                # 卷积之后，如果接BN操作，就不要设置偏置，因为不起作用还要占显卡内存。
                #nn.BatchNorm2d(out_ch),
                nn.GELU(),
            )


    def forward(self, x):

        x_M_N = self.M_N(x)             # 跳跃连接
        x_1 =  self.N_1(x)              # 得到通道数为self.number的特征图
        x_3_0 = self.N_N_3_3(x_1)
        x_3_1 = self.N_N_3_3(x_3_0)
        x_3_2 = self.N_N_3_3(x_3_1)
        x_3_3 = self.N_N_3_3(x_3_2)

        x =  torch.cat((x_3_0, x_3_1, x_3_2 , x_3_3), 1)

        y = self.one_one(x)

        y = x_M_N + y

        return F.gelu(y)

class down_one(nn.Module):  # 完成一个下采样的过程，第一列的结果 输出范围为  0---+无穷大
    def __init__(self, in_ch, out_ch, BN=True,only_MAX=False,num = 0 , TRB_or_DOWN = False):
        super(down_one, self).__init__()
        self.BN = BN
        self.only_MAX = only_MAX
        self.conv0 = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1, padding=0, bias=True)  # 输入到中间层
        self.conv0_down = nn.Conv2d(1, in_ch, kernel_size=3, stride=2, padding=1, bias=True)  # 输入到中间层
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.out = nn.Conv2d(2*in_ch, out_ch, kernel_size=3, stride=1, padding=1,bias=True)
        self.conv_max = nn.Conv2d(1, out_ch, kernel_size=1, stride=1, padding=0, bias=True)  # 输入到中间层
        self.TRB_or_Down = TRB_or_DOWN
        if self.TRB_or_Down:
            self.TRB_num = num
            self.TRB = TRBlock(out_ch, out_ch)
        else:
            self.DOWN_num = num
            self.DOWN = DOWNlock(out_ch, out_ch)

    def forward(self, inputs):
        if self.only_MAX:
            inputs = self.conv0(inputs)
            inputs = self.conv_max(inputs)
            x_out = self.maxpool(inputs)
        else:
            x_conv = self.conv0(inputs)
            x_conv_down = self.conv0_down(x_conv)
            x_maxpool = self.maxpool(inputs)
            x_cat = torch.cat((x_conv_down, x_maxpool), 1)
            x_out = self.out(x_cat)
        if self.TRB_or_Down:
            for i in range(0, self.TRB_num):
                x_out = self.TRB(x_out)
        else:
            for i in range(0, self.DOWN_num):
                x_out = self.DOWN(x_out)

        return  x_out

#  self.up_concat10 = unetUp_origin_me(filters[1], filters[0], self.is_deconv,num = self.num)
class unetUp_origin_me(nn.Module):   # 实现一个上卷积 （1，in_size,w，h）->（1，out_size，2w,2h）
    def __init__(self, in_size, out_size, is_deconv, n_concat=2 , num = 1):         # 输入通道数，输出通道数
        super(unetUp_origin_me, self).__init__()
        self.canchakuai = Canchakuai_me(out_size, out_size, BN = True,shortcut=None)
        self.num = num
        if is_deconv:  # Ture
            self.conv = unetConv2_2(in_size + (n_concat - 2) * out_size, out_size, False)         # 两次卷积，尺寸不变
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)   # 两次反卷积，尺寸发生变化  （1，in_size,w，h）->（1，out_size，2w,2h）
        else:
            self.conv = unetConv2_2(in_size + (n_concat - 2) * out_size, out_size, False)         # 两次卷积，尺寸不变
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)                                     #与转置卷积相比,双线性插值上采样计算相对简单,但生成质量可能不如学习的转置卷积。实际应用中需要权衡计算开销和生成质量的tradeoff来选择合适的上采样方式。

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue    # 跳过名字包含unetConv2名称的模块
            init_weights(m, init_type='kaiming')                         # 对其他模块进行kaiming初始化

    def forward(self,inputs0,input):
        outputs0 = self.up(inputs0)                                      # 进行上采样
        for i in range(0,self.num):
            input = self.canchakuai(input)
        outputs = torch.cat([outputs0,input], 1)
        return self.conv(outputs)
class unetUp_origin_end_me(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2,BN =True,num = 1):
        super(unetUp_origin_end_me, self).__init__()
        self.BN = BN
        self.num = num
        self.canchakuai = Canchakuai_me(out_size, out_ch=out_size, BN=True, shortcut=None)
        if is_deconv:                    # True
            self.conv = unetConv2_com(in_size + (n_concat - 2) * out_size, out_size, False,BN=self.BN)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConv2_com(in_size + (n_concat - 2) * out_size, out_size, False,BN=self.BN)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, input):
        outputs0 = self.up(inputs0)
        for i in range(0, self.num):
            input = self.canchakuai(input)
        outputs = torch.cat([outputs0, input], 1)
        return self.conv(outputs)


class MSUNet_me_3(nn.Module):
    def __init__(self, in_channels=3, n_classes=3, feature_scale=4, is_deconv=True,
                 is_batchnorm=True, is_ds=True, BN =True, num= 1,only_MAX = False
                 ,TRB_or_DOWN_num = 1,TRB_or_DOWN=False):
        super(MSUNet_me_3, self).__init__()
        self.is_deconv = is_deconv                # True
        self.in_channels = in_channels            # 3
        self.is_batchnorm = is_batchnorm          # True
        self.is_ds = is_ds                        # True
        self.feature_scale = feature_scale        # 4
        self.CatChannels = 64
        self.BN = BN
        self.num = num
        self.only_MAX = only_MAX
        self.TRB_or_DOWN_num =TRB_or_DOWN_num
        filters = [32, 64, 128, 256, 512]   # 每层输出

        # 设计
        # 温度损失
        self.T = self.T_loss
        # -------------------下采样---------------------------
        # 第一列下采样
        self.conv00 = nn.Conv2d(self.in_channels, filters[0], kernel_size=1, stride=1, padding=0, bias=True)
        self.conv01 = down_one(filters[0], filters[1], BN=self.BN,only_MAX=self.only_MAX,num=self.TRB_or_DOWN_num,TRB_or_DOWN =TRB_or_DOWN)
        self.conv02 = down_one(filters[1], filters[2], BN=self.BN,only_MAX=self.only_MAX,num=self.TRB_or_DOWN_num,TRB_or_DOWN =TRB_or_DOWN)
        self.conv03 = down_one(filters[2], filters[3], BN=self.BN,only_MAX=self.only_MAX,num=self.TRB_or_DOWN_num,TRB_or_DOWN =TRB_or_DOWN)
        self.conv04 = down_one(filters[3], filters[4], BN=self.BN,only_MAX=self.only_MAX,num=self.TRB_or_DOWN_num,TRB_or_DOWN =TRB_or_DOWN)
        # -------------------下采样---------------------------

        # -------------------上采样---------------------------
        # 第二列上采样
        self.up_concat10 = unetUp_origin_me(filters[1], filters[0], self.is_deconv,n_concat=2,num = self.num)
        self.up_concat11 = unetUp_origin_me(filters[2], filters[1], self.is_deconv,n_concat=2,num = self.num)
        self.up_concat12 = unetUp_origin_me(filters[3], filters[2], self.is_deconv,n_concat=2,num = self.num)
        self.up_concat13 = unetUp_origin_end_me(filters[4], filters[3], self.is_deconv, BN=self.BN,n_concat=2,num = self.num)  # 添加注意力模块


        # 第最后一行中残差模块
        self.h03_PT_hd13 = Canchakuai_me(in_ch=filters[3], out_ch=filters[3], BN=True, shortcut=None)

        # 第三列上采样
        self.up_concat20 = unetUp_origin_me(filters[1], filters[0], self.is_deconv, n_concat=2,num = self.num)
        self.up_concat21 = unetUp_origin_me(filters[2], filters[1], self.is_deconv, n_concat=2,num = self.num)
        self.up_concat22 = unetUp_origin_end_me(filters[3], filters[2], self.is_deconv, n_concat=2, BN=self.BN,num = self.num)  # 添加注意力模块
        # 第四列上采样
        self.up_concat30 = unetUp_origin_me(filters[1], filters[0], self.is_deconv, n_concat=2,num = self.num)
        self.up_concat31 = unetUp_origin_end_me(filters[2], filters[1], self.is_deconv, n_concat=2, BN=self.BN,num = self.num)  # 添加注意力模块
        # 第五列上采样
        self.up_concat40 = unetUp_origin_end_me(filters[1], filters[0], self.is_deconv, n_concat=2, BN=self.BN,num = self.num)  # 添加注意力模块
        # -------------------上采样---------------------------
        # -------------------最后一层的最大池化融合--------------
        # 完成13的设计
        self.h00_PT_hd13 = nn.MaxPool2d(8, 8,ceil_mode=True)  # ceil_mode=True 表示输出特征图的大小将向上取整 (1,a,w,h)->(1,a,w/8,h/8) 并且向下取整
        self.h00_PT_hd13_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)  # 卷积 （1，32，w，h）  ->  （1，64，w，h）
        self.h00_PT_hd13_bn = nn.BatchNorm2d(self.CatChannels)  # 归一化 64个通道
        self.h00_PT_hd13_relu = nn.GELU()  # 激活层

        self.h01_PT_hd13 = nn.MaxPool2d(4, 4,ceil_mode=True)  # ceil_mode=True 表示输出特征图的大小将向上取整 (1,a,w,h)->(1,a,w/4,h/4) 并且向下取整
        self.h01_PT_hd13_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)  # 卷积 （1，64，w，h）  ->  （1，64，w，h）
        self.h01_PT_hd13_bn = nn.BatchNorm2d(self.CatChannels)  # 归一化 64个通道
        self.h01_PT_hd13_relu = nn.GELU()  # 激活层

        self.h02_PT_hd13 = nn.MaxPool2d(2, 2,ceil_mode=True)  # ceil_mode=True 表示输出特征图的大小将向上取整 (1,a,w,h)->(1,a,w/2,h/2) 并且向下取整
        self.h02_PT_hd13_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)  # 卷积 （1，128，w，h）  ->  （1，64，w，h）
        self.h02_PT_hd13_bn = nn.BatchNorm2d(self.CatChannels)  # 归一化 64个通道
        self.h02_PT_hd13_relu = nn.GELU()  # 激活层
        # 残差连接并且改变通道数，改变通道数，可以添加多个进行残差连接
        #self.h03_PT_hd13 = Canchakuai_me(in_ch=filters[3], out_ch=filters[3], BN = True,shortcut=None)

        # self.conv13 = nn.Conv2d(448, 256, 1)
        self.conv13 = nn.Sequential(nn.Conv2d(filters[3]+self.CatChannels+self.CatChannels+self.CatChannels, filters[3], 1),  # 改变通道数
                                    nn.BatchNorm2d(filters[3]),
                                    nn.GELU(), )

        # 完成22的设计
        self.h00_PT_hd22 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h00_PT_hd22_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h00_PT_hd22_bn = nn.BatchNorm2d(self.CatChannels)
        self.h00_PT_hd22_relu = nn.GELU()

        self.h01_PT_hd22 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h01_PT_hd22_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h01_PT_hd22_bn = nn.BatchNorm2d(self.CatChannels)
        self.h01_PT_hd22_relu = nn.GELU()
        # 残差连接并且改变通道数，改变通道数，可以添加多个进行残差连接
        #self.h02_PT_hd22 = Canchakuai_me(in_ch=filters[2], out_ch=filters[2], BN=True, shortcut=None)
        # self.conv22 = nn.Conv2d(384, 128, 1)
        self.conv22 = nn.Sequential(nn.Conv2d(filters[2]+self.CatChannels+self.CatChannels+filters[2],filters[2], 1),
                                    nn.BatchNorm2d(filters[2]),
                                    nn.GELU(), )
        # 完成31的设计
        self.h00_PT_hd31 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h00_PT_hd31_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h00_PT_hd31_bn = nn.BatchNorm2d(self.CatChannels)
        self.h00_PT_hd31_relu = nn.GELU()
        # 残差连接并且改变通道数，改变通道数，可以添加多个进行残差连接
        #self.h01_PT_hd31 = Canchakuai_me(in_ch=filters[1], out_ch=filters[1], BN=True, shortcut=None)
        # self.conv31 = nn.Conv2d(192, 64, 1)
        self.conv31 = nn.Sequential(nn.Conv2d(filters[1]+self.CatChannels+filters[1], filters[1], 1),
                                    nn.BatchNorm2d(filters[1]),
                                    nn.GELU(), )
        # 完成40的设计
        # 残差连接并且改变通道数，改变通道数，可以添加多个进行残差连接
        self.h00_PT_hd40 = Canchakuai_me(in_ch=filters[0], out_ch=filters[0], BN=True, shortcut=None)
        # self.conv40 = nn.Conv2d(64, 32, 1)
        self.conv40 = nn.Sequential(nn.Conv2d(filters[0] + filters[0], filters[0], 1),
                                    nn.BatchNorm2d(filters[0]),
                                    nn.GELU(), )
        # 最上面的四个输出
        # final conv (without any concat)  将通道数降为输出的三层,最上面四个输出
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        # 设计
        # initialise weights  初始化网络
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
    def T_loss(self, input): # 温度损失，输入一个灰度图像进行边缘检测
        # 计算x方向的边缘特征
        sobelx = cv2.Sobel(input, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)

        # 计算y方向的边缘特征
        sobely = cv2.Sobel(input, cv2.CV_64F, 0, 1, ksize=3)
        sobely = cv2.convertScaleAbs(sobely)

        # 取反x方向的边缘特征
        sobelx_inverted = cv2.bitwise_not(sobelx)

        # 取反y方向的边缘特征
        sobely_inverted = cv2.bitwise_not(sobely)

        # 融合x,y方向的反边缘特征
        edges_inverted = cv2.addWeighted(sobelx_inverted, 0.5, sobely_inverted, 0.5, 0)

        # Otsu处理
        threshold_T_otsu, threshold_otsu = cv2.threshold(input, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 将边缘检测的结果与otus的结果进行融合

        total = cv2.addWeighted(edges_inverted, 0.8, threshold_otsu, 0.2, 0)

        total_ir = cv2.addWeighted(total, 0.05, input, 0.95, 0)

        return total_ir

    def forward(self, inputs):

        # 设计
        # 得到第一列的输出
        X_00 = self.conv00(inputs)  # （1，3，w，h）  ->  （1，32，w，h）
        X_01 = self.conv01(X_00)  # （1，32，w，h）  ->  （1，64，w/2，h/2）
        X_02 = self.conv02(X_01)  # （1，64，w/2，h/2）  ->  （1，128，w/4，h/4）
        X_03 = self.conv03(X_02)  # （1，128，w/4，h/4）  ->  （1，256，w/8，h/8）
        X_04 = self.conv04(X_03)  # （1，256，w/8，h/8）  ->  （1，512，w/16，h/16）
        # 得到第二列的输出
        X_10 = self.up_concat10(X_01, X_00)  # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，32，w，h）
        X_11 = self.up_concat11(X_02, X_01)  # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，64，w/2，h/2）
        X_12 = self.up_concat12(X_03, X_02)  # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，128，w/4，h/4）
        # 此处可以设计一个14
        # 此处可以设计一个14
        X_13 = self.up_concat13(X_04, self.h03_PT_hd13(X_03))  # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，256，w/8，h/8）

        # 得到13的输出
        h00_hd13 = self.h00_PT_hd13_relu(self.h00_PT_hd13_bn(self.h00_PT_hd13_conv(self.h00_PT_hd13(X_00))))
        h01_hd13 = self.h01_PT_hd13_relu(self.h01_PT_hd13_bn(self.h01_PT_hd13_conv(self.h01_PT_hd13(X_01))))
        h02_hd13 = self.h02_PT_hd13_relu(self.h02_PT_hd13_bn(self.h02_PT_hd13_conv(self.h02_PT_hd13(X_02))))
        X_13 = torch.cat((X_13, h00_hd13, h01_hd13, h02_hd13), 1)  # 256+64+64+64= 448

        X_13 = self.conv13(X_13)  # （1，448，w/8，h/8）  ->  （1，256，w/8，h/8）
        # 第二列的输出
        X_20 = self.up_concat20(X_11, X_10)  # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，32，w，h）
        X_21 = self.up_concat21(X_12, X_11)  # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，64，w/2，h/2）
        X_22 = self.up_concat22(X_13, X_12)  # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，128，w/4，h/4）
        # 得到22的输出
        h00_hd22 = self.h00_PT_hd22_relu(self.h00_PT_hd22_bn(self.h00_PT_hd22_conv(self.h00_PT_hd22(X_00))))
        h01_hd22 = self.h01_PT_hd22_relu(self.h01_PT_hd22_bn(self.h01_PT_hd22_conv(self.h01_PT_hd22(X_01))))
        h02_hd22 = self.h02_PT_hd22(X_02)
        X_22 = torch.cat((X_22,h00_hd22, h01_hd22,h02_hd22), 1)  # 128+64+64 =256
        X_22 = self.conv22(X_22)  # （1，256，w/4，h/4）  ->  （1，128，w/4，h/4）
        # 第三列输出
        X_30 = self.up_concat30(X_21, X_20)  # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，32，w，h）
        X_31 = self.up_concat31(X_22, X_21)  # 第一个为需要上采样的（尺寸变大两倍）   输出：（1，64，w/2，h/2）
        h00_hd31 = self.h00_PT_hd31_relu(self.h00_PT_hd31_bn(self.h00_PT_hd31_conv(self.h00_PT_hd31(X_00))))
        h01_hd31 = self.h01_PT_hd31(X_01)
        X_31 = torch.cat((X_31, h00_hd31,h01_hd31), 1)  # 64+64 = 128
        X_31 = self.conv31(X_31)  # （1，128，w/2，h/2）  ->  （1，64，w/2，h/2）
        # 第四列输出
        X_40 = self.up_concat40(X_31, X_30)  # (1,32，w，h)
        h00_hd40 = self.h00_PT_hd40(X_00)
        X_40 = torch.cat((X_40, h00_hd40), 1)  # 64+64 = 128
        X_40 = self.conv40(X_40)  # （1，128，w/2，h/2）  ->  （1，64，w/2，h/2）

        # final layer  将最上面的四个输出进行3通道的变换
        final_1 = self.final_1(X_10)
        final_2 = self.final_2(X_20)
        final_3 = self.final_3(X_30)
        final_4 = self.final_4(X_40)

        final = (final_1 + final_2 + final_3 + final_4) / 4  # 将取得的彩色化结果进行最上面的4个输出进行平均叠加

        if self.is_ds:
            return F.leaky_relu(final,0.2)                         # 适当的保存负值避免梯度消失
        else:
            return F.leaky_relu(final_4,0.2)
        # 设计
