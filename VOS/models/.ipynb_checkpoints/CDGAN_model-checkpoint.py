from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import os

import cv2
import numpy as np
import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from models.vgg import Vgg16

import torch
from MS_SSIM import ssim
from torchstat import stat
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as compare_ssim

from skimage import io
import lpips

from util import util




class CDGANModel(BaseModel):
    def name(self):
        return 'CDGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # parser.set_defaults(dataset_mode='aligned')
        # parser.set_defaults(which_model_netG='unet_256')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G1_GAN','G2_GAN','vgg_G1','vgg_G2','ssim_G1','ssim_G2', 'a', 'b','a_step','b_step', 'G1', 'G2','D1','D2']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_Aa','fake_a', 'real_Ba','real_Ab','fake_b','real_Bb','real_A','fake_AB','real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G1', 'G2', 'D1','D2',]
        else:  # during test time, only load Gs
            self.model_names = ['G1','G2']
        # use_gan
        self.use_gan = opt.use_GAN
        self.w_vgg = opt.w_vgg  # vgg 网络的比重 1
        self.w_tv = opt.w_tv    # tv 损失的比重 1.25
        self.w_gan = opt.w_gan  # gan网络的损失比重 0.03
        self.w_ss = opt.w_ss    # ms-ssim损失的比重 1.25
        self.w_ab = opt.w_ab    # 分割像素损失的比重 1
        self.w_step = opt.w_step # 步长损失的比重 5
        self.use_condition = opt.use_condition  # 使用condition（条件，提前）来进行 1
        # load/define networks  (输入图像通道-3，输出图像通道-3，第一个卷积层的生成过滤器（层数）-32，选择那个模型作为生成器-MSUNet，选择归一化方法-instance，是否使用dropout层—使用，网络初始化-normal（正态分布初始化）,设备号-0)

        self.netG1 = networks.define_G1(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG1, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        self.netG2 = networks.define_G2(opt.input_nc, opt.output_nc, opt.ngf,
                                        opt.which_model_netG2, opt.norm, not opt.no_dropout, opt.init_type,
                                        self.gpu_ids)


        if self.isTrain:
            use_sigmoid = opt.no_lsgan # 是否使用标准GAN还是最小二乘GAN，use_sigmoid = 0
            if self.use_condition == 1:   #选择使用的不同鉴别器
                # (输入图像通道-6，第一个卷积层的生成过滤器（层数）-32，
                # 选择那个模型作为生成器-basic，
                # 仅仅当which_model_netD==n_layers时使用（这里不使用），选择归一化方法-instance，是否使用标准GAN还是最小二乘GAN-0，网络初始化-normal（正态分布初始化）,设备号-0)
                self.netD1 = networks.define_D1(opt.input_nc + opt.output_nc, opt.ndf,
                                              opt.which_model_netD1,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
                self.netD2 = networks.define_D2(opt.input_nc + opt.output_nc, opt.ndf,
                                                opt.which_model_netD2,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            else:
                # (输入图像通道-3，第一个卷积层的生成过滤器（层数）-32，
                # 选择那个模型作为生成器-basic，
                # 仅仅当which_model_netD==n_layers时使用（这里不使用），选择归一化方法-instance，是否使用标准GAN还是最小二乘GAN-0，网络初始化-normal（正态分布初始化）,设备号-0)
                self.netD1 = networks.define_D1(opt.input_nc, opt.ndf,
                                              opt.which_model_netD1,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
                self.netD2 = networks.define_D1(opt.input_nc, opt.ndf,
                                                opt.which_model_netD2,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)


        if self.isTrain:
            # 创建一个ImagePool对象,用于存储先前生成的图像,该ImagePool对象的大小或容量为50。
            self.fake_Aa_pool = ImagePool(opt.pool_size)  #  opt.pool_size = 50 存储先前生成图像的图像缓冲区的大小
            self.fake_Ab_pool = ImagePool(opt.pool_size)  # opt.pool_size = 50 存储先前生成图像的图像缓冲区的大小
            # define loss functions
            if opt.which_model_netD1 == 'multi':
                self.criterionGAN = networks.GANLoss_multi(use_lsgan=not opt.no_lsgan).to(self.device)
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=opt.no_lsgan).to(self.device)  #原本是use_lsgan=not opt.no_lsgan  ，这里use_lsgan = 0 这里使用BCE损失
            self.criterionL1 = torch.nn.L1Loss() # 这是采用L1损失来计算损失
            self.criterionL2 = torch.nn.MSELoss()  # 这是采用MSE损失来计算损失

            # load vgg network
            self.vgg = Vgg16().type(torch.cuda.FloatTensor) # 上文中已经定义 from models.vgg import Vgg16 确定文件位置

            # initialize optimizers
            self.optimizers = []
            # betas=(opt.beta1, 0.999): 这里设置了 Adam 优化器的 beta 参数。beta1 控制动量(momentum)的大小,通常取 0.9。beta2 控制估计二阶矩的指数衰减率,通常取 0.999。 ---0.5
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G1)
            self.optimizers.append(self.optimizer_D1)
            self.optimizers.append(self.optimizer_G2)
            self.optimizers.append(self.optimizer_D2)
    def set_input(self, input):  #
        AtoB = self.opt.which_direction == 'AtoB'

        self.real_a = input['a' if AtoB else 'b'].to(self.device)  # 红外图像a
        self.real_b = input['b' if AtoB else 'a'].to(self.device)  # 图像b
        self.real_Ba = input['Ba' if AtoB else 'Bb'].to(self.device)  # 真实图像B中的a
        self.real_Bb = input['Bb' if AtoB else 'Ba'].to(self.device)  # 真实图像B中的b
        self.real_A = input['A' if AtoB else 'B'].to(self.device)  # 图像A
        self.real_B = input['B' if AtoB else 'A'].to(self.device)  # 图像B
        self.real_Aa = input['Aa' if AtoB else 'Ab'].to(self.device)  # 红外图像A中的图像a
        self.real_Ab = input['Ab' if AtoB else 'Aa'].to(self.device)  # 红外图像B中的图像b

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self, l1, l2,best_ls_x,d_ls):
        self.fake_a = self.netG1(self.real_a)
        self.fake_b = self.netG2(self.real_b)
        self.d_ls = d_ls

        #----------------------可视化----------------------------

        # img_fake_a = util.tensor2im(self.fake_a)
        # img_fake_b = util.tensor2im(self.fake_b)
        # img_real_a = util.tensor2im(self.real_a)
        # img_real_b = util.tensor2im(self.real_b)

        # cv2.imshow('1', img_fake_a)    # 生成的a
        # cv2.imshow('2', img_fake_b)    # 生成的b
        # cv2.imshow('3', img_real_a)    # 红外的a
        # cv2.imshow('4', img_real_b)    # 红外的b
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # ----------------------可视化----------------------------


        # 分割求出 fake_a_I,fake_a_II,fake_b_I,fake_b_II
        # 裁剪为

        self.fake_a_I = self.fake_a[0:1,0:3,0:256,0:128]
        #print('self.fake_a_I.shape:',self.fake_a_I.shape)
        #print('self.fake_a.shape:', self.fake_a.shape)
        self.fake_a_II = self.fake_a[0:1,0:3,0:256,128:256]
        #print('self.fake_a_II.shape:',self.fake_a_II.shape)
        self.fake_b_II = self.fake_b[0:1,0:3,0:256,0:128]
        #print('self.fake_b_II.shape:',self.fake_b_II.shape)
        self.fake_b_III = self.fake_b[0:1,0:3,0:256,128:256]
        #print('self.fake_b_III.shape:',self.fake_b_III.shape)
        #print('self.real_Ba.shape:', self.real_Ba.shape)
        self.real_Ba_I = self.real_Ba[0:1,0:3,0:256,0:128]
        # print('self.real_Ba_I.shape:', self.real_Ba_I.shape)
        self.real_Bab_II = self.real_Ba[0:1,0:3,0:256,128:256]
        # print('self.real_Bab_II.shape:', self.real_Bab_II.shape)
        self.real_Bb_III = self.real_Bb[0:1,0:3,0:256,128:256]
        # print('self.real_Bb_III.shape:', self.real_Bb_III.shape)
        # 步长损失
        self.fake_a_step = self.fake_a[0:1,0:3,0:256,256-best_ls_x:256]
        self.fake_b_step = self.fake_b[0:1, 0:3, 0:256, 0:best_ls_x]
        self.real_Ba_step = self.real_Ba[0:1,0:3,0:256,256-best_ls_x:256]
        self.real_Bb_step = self.real_Bb[0:1, 0:3, 0:256, 0:best_ls_x]
        # ------------------可视化---------------------------------

        # img_fake_a = util.tensor2im(self.fake_a)
        # img_fake_b = util.tensor2im(self.fake_b)
        # img_real_a = util.tensor2im(self.real_a)
        # img_real_b = util.tensor2im(self.real_b)
        # img_fake_a_I = util.tensor2im(self.fake_a_I)
        # img_fake_a_II = util.tensor2im(self.fake_a_II)
        # img_fake_b_II = util.tensor2im(self.fake_b_II)
        # img_fake_b_III = util.tensor2im(self.fake_b_III)
        # img_real_Ba_I = util.tensor2im(self.real_Ba_I)
        # img_real_Bab_II = util.tensor2im(self.real_Bab_II)
        # img_real_Bb_III = util.tensor2im(self.real_Bb_III)
        # img_real_B = util.tensor2im(self.real_B)
        # img_real_Ba = util.tensor2im(self.real_Ba)
        # img_real_Bb = util.tensor2im(self.real_Bb)
        #
        # cv2.imshow('img_fake_a', img_fake_a)  # 生成的a
        # cv2.imshow('img_fake_b', img_fake_b)  # 生成的b
        # cv2.imshow('img_real_a', img_real_a)  # 红外的a
        # cv2.imshow('img_real_b', img_real_b)  # 红外的b
        # cv2.imshow('img_fake_a_I', img_fake_a_I)  # 生成的a
        # cv2.imshow('img_fake_a_II', img_fake_a_II)  # 生成的b
        # cv2.imshow('img_fake_b_II', img_fake_b_II)  # 红外的a
        # cv2.imshow('img_fake_b_III', img_fake_b_III)  # 红外的b
        # cv2.imshow('img_real_Ba_I', img_real_Ba_I)  # 生成的a
        # cv2.imshow('img_real_Bab_II', img_real_Bab_II)  # 生成的b
        # cv2.imshow('img_real_Bb_III', img_real_Bb_III)  # 红外的a
        # cv2.imshow('img_real_B', img_real_B)  # 红外的a
        # cv2.imshow('img_real_Ba', img_real_Ba)  # 红外的a
        # cv2.imshow('img_real_Bb', img_real_Bb)  # 红外的a
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # ------------------可视化---------------------------------

        # 求相交区域的图像
        self.fake_ab_II = (self.fake_a_II+self.fake_b_II)/2

        self.l1 = l1
        self.l2 = l2


        if l1 == 0:
            self.fake_ab_II_np = self.fake_ab_II[0].cpu().detach().numpy()  # 三维
            self.fake_b_III_np = self.fake_b_III[0].cpu().detach().numpy()

            self.fake_ab_II_np = np.transpose(self.fake_ab_II_np, (1, 2, 0))
            self.fake_b_III_np = np.transpose(self.fake_b_III_np, (1, 2, 0))

            self.fake_ab_II_np = cv2.resize(self.fake_ab_II_np, (l2-l1, 256))
            self.fake_b_III_np = cv2.resize(self.fake_b_III_np, (320-l2, 256))

            self.fake_AB = np.concatenate((self.fake_ab_II_np,self.fake_b_III_np), axis=1)   # 拼接操作 需要先将数据转换为cpu格式

        elif l2 == 320:
            self.fake_a_I_np = self.fake_a_I[0].cpu().detach().numpy()
            self.fake_ab_II_np = self.fake_ab_II[0].cpu().detach().numpy()  # 三维

            self.fake_a_I_np = np.transpose(self.fake_a_I_np, (1, 2, 0))
            self.fake_ab_II_np = np.transpose(self.fake_ab_II_np, (1, 2, 0))

            self.fake_a_I_np = cv2.resize(self.fake_a_I_np, (l1, 256))
            self.fake_ab_II_np = cv2.resize(self.fake_ab_II_np, (l2-l1, 256))

            self.fake_AB = np.concatenate((self.fake_a_I_np, self.fake_ab_II_np), axis=1)  # 拼接操作 需要先将数据转换为cpu格式

        else:
            self.fake_a_I_np = self.fake_a_I[0].cpu().detach().numpy()
            self.fake_ab_II_np = self.fake_ab_II[0].cpu().detach().numpy()  # 三维
            self.fake_b_III_np = self.fake_b_III[0].cpu().detach().numpy()

            self.fake_a_I_np = np.transpose(self.fake_a_I_np, (1, 2, 0))
            self.fake_ab_II_np = np.transpose(self.fake_ab_II_np, (1, 2, 0))
            self.fake_b_III_np = np.transpose(self.fake_b_III_np, (1, 2, 0))

            self.fake_a_I_np = cv2.resize(self.fake_a_I_np, (l1, 256))
            self.fake_ab_II_np = cv2.resize(self.fake_ab_II_np, (l2-l1, 256))
            self.fake_b_III_np = cv2.resize(self.fake_b_III_np, (320-l2, 256))

            self.fake_AB = np.concatenate((self.fake_a_I_np, self.fake_ab_II_np, self.fake_b_III_np),axis=1)  # 拼接操作 需要先将数据转换为cpu格式


        self.fake_AB = transforms.ToTensor()(self.fake_AB).to(self.device)  # (0 - 255) --> (0 - 1)  得到的数据类型为（3,256,320）
        self.fake_AB = self.fake_AB.unsqueeze(0)  # 给矢量数组增添一个维度



    def backward_D1(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # 从之前生成的假样本中选择一些样本,将它们与当前的真实样本拼接起来,形成一个新的输入样本。这个新的输入样本可能会被用于后续的网络训练或评估。
        # query() 函数通过维护一个假样本池,并从中选择样本用于训练,在提高 GAN 训练稳定性、生成样本质量等方面发挥了重要作用。
        if self.use_condition == 1:
            fake_AB = self.fake_Aa_pool.query(torch.cat((self.real_Aa, self.fake_a), 1))
        else:
            fake_AB = self.fake_a
        pred_fake = self.netD1(fake_AB.detach()) # .detach()分离新的数据，使其不影响之前的值
        # loss = -(1 - y) * log(1 - pred) - y * log(pred)  格式为（pred，T/F）
        self.loss_D1_fake = self.criterionGAN(pred_fake, False)    # 鉴别器计算BCE损失，False代表标签时 自己手动打标签  loss = -log(1 - sigmoid(pred))  应该趋近于0

        # Real
        if self.use_condition == 1:
            real_AB = torch.cat((self.real_Aa, self.real_Ba), 1)
        else:
            real_AB = self.real_Ba
        pred_real = self.netD1(real_AB)
        self.loss_D1_real = self.criterionGAN(pred_real, True)     # 鉴别器计算BCE损失，True代表标签 自己手动打标签 loss = log(sigmoid(pred))   应该趋近于1

        # Combined loss
        self.loss_D1 = (self.loss_D1_fake + self.loss_D1_real) * 0.5  # 将两个损失进行平均输出

        self.loss_D1.backward()

    def backward_G1(self):
        # First, G(A) should fake the discriminator
        if self.use_gan == 1:
            if self.use_condition == 1:  # 采用
                fake_AB = torch.cat((self.real_Aa, self.fake_a), 1) # 将两个图像进行cat在通道层进行叠加
            else:
                fake_AB = self.fake_a
            pred_fake = self.netD1(fake_AB)
            self.loss_G1_GAN = self.criterionGAN(pred_fake, True)  # 鉴别器计算BCE损失，True代表标签 自己手动打标签 loss = log(pred)
        else:
            self.loss_G1_GAN = 0

        # Second, G(A) = B
        # self.loss_G1_L1 = self.criterionL1(self.fake_a, self.real_Ba)   # 生成器的L1损失
        # vgg loss 四个特征层的信息进行L1损失
        self.real_Ba_features = self.vgg(self.real_Ba)
        self.fake_Ba_features = self.vgg(self.fake_a)
        self.loss_vgg_G1 = self.criterionL1(self.fake_Ba_features[1], self.real_Ba_features[1]) * 1 + \
                        self.criterionL1(self.fake_Ba_features[2], self.real_Ba_features[2]) * 1 + \
                        self.criterionL1(self.fake_Ba_features[3],self.real_Ba_features[3]) * 1 + \
                        self.criterionL1(self.fake_Ba_features[0], self.real_Ba_features[0]) * 1
        # # TV loss 固定公式，只和生成的图片有关系，即fake_B
        # # torch.abs(self.fake_B[:, :, :, 1:] - self.fake_B[:, :, :, :-1]): 这个操作是计算上面两步获取的两个张量之间的元素级绝对差值。这样做可以得到相邻像素在垂直方向上的绝对差值。
        # # 计算的目的通常是用来作为一种正则化项,用于鼓励生成器网络生成平滑的图像。
        # diff_i = torch.sum(torch.abs(self.fake_a[:, :, :, 1:] - self.fake_a[:, :, :, :-1]))  # 列
        # diff_j = torch.sum(torch.abs(self.fake_a[:, :, 1:, :] - self.fake_a[:, :, :-1, :]))  # 行
        # self.tv_loss_G1 = (diff_i + diff_j) / (320 * 256)  # 因为此时文章提取到的图像是320×256的尺寸
        #
        # ssim-loss 设计ssim损失进行约束函数
        # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
        X = ((self.real_Ba + 1) / 2)
        Y = ((self.fake_a + 1) / 2)
        #
        self.loss_ssim_G1 = 1 - ssim(X, Y, data_range=1, size_average=True)  # data_range=1 输入范围为 0-1 size_average=True 输入每个值的平均值 loss_ssim ：越低越真

        # 分割像素损失

        if self.l1 ==0:   # 此时没有a_I的分割损失
            self.loss_a_II = self.criterionL1(self.fake_a_II, self.real_Bab_II)   # a_II的分割损失
            self.loss_a = self.loss_a_II
        else:
            self.loss_a_I = self.criterionL1(self.fake_a_I, self.real_Ba_I)  # a_I的分割损失
            self.loss_a_II = self.criterionL1(self.fake_a_II, self.real_Bab_II)  # a_II的分割损失
            self.loss_a = self.loss_a_I + self.loss_a_II

        # 步长损失

        if self.l1 == 0 or self.l2 == 320 or self.d_ls<=0:  # 此时没有步长损失
            self.loss_a_step = 0
        else:
            self.loss_a_step = self.criterionL2(self.fake_a_step, self.real_Ba_step)


        # loss_G = loss_G_GAN × 0.03 + loss_G_L1 × 1 + loss_vgg × 1 + tv_loss × 1.25 + loss_ssim × 1.25 + loss_a_step × 1.5
        # self.loss_G1 = self.loss_G1_GAN * self.w_gan + self.loss_G1_L1 + self.loss_vgg_G1 * self.w_vgg + self.tv_loss_G1 * self.w_tv + self.w_ss * self.loss_ssim_G1 + self.loss_a*self.w_ab
        # self.loss_G1 = self.loss_G1_GAN * self.w_gan + self.loss_vgg_G1 * self.w_vgg + self.tv_loss_G1 * self.w_tv + self.w_ss * self.loss_ssim_G1 + self.loss_a * self.w_ab + self.loss_a_step * self.w_step
        self.loss_G1 = self.loss_G1_GAN * self.w_gan + self.loss_vgg_G1 * self.w_vgg + self.w_ss * self.loss_ssim_G1 + self.loss_a * self.w_ab + self.loss_a_step * self.w_step

        self.loss_G1.backward() # 反向传播

    def backward_D2(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # 从之前生成的假样本中选择一些样本,将它们与当前的真实样本拼接起来,形成一个新的输入样本。这个新的输入样本可能会被用于后续的网络训练或评估。
        # query() 函数通过维护一个假样本池,并从中选择样本用于训练,在提高 GAN 训练稳定性、生成样本质量等方面发挥了重要作用。
        if self.use_condition == 1:
            fake_AB = self.fake_Ab_pool.query(torch.cat((self.real_Ab, self.fake_b), 1))
        else:
            fake_AB = self.fake_b
        pred_fake = self.netD2(fake_AB.detach()) # .detach()分离新的数据，使其不影响之前的值
        # loss = -(1 - y) * log(1 - pred) - y * log(pred)  格式为（pred，T/F）
        self.loss_D2_fake = self.criterionGAN(pred_fake, False)    # 鉴别器计算BCE损失，False代表标签时 自己手动打标签  loss = -log(1 - sigmoid(pred))  应该趋近于0

        # Real
        if self.use_condition == 1:
            real_AB = torch.cat((self.real_Ab, self.real_Bb), 1)
        else:
            real_AB = self.real_Bb
        pred_real = self.netD2(real_AB)
        self.loss_D2_real = self.criterionGAN(pred_real, True)     # 鉴别器计算BCE损失，True代表标签 自己手动打标签 loss = log(sigmoid(pred))   应该趋近于1

        # Combined loss
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5  # 将两个损失进行平均输出

        self.loss_D2.backward()

    def backward_G2(self):
        # First, G(A) should fake the discriminator
        if self.use_gan == 1:
            if self.use_condition == 1:  # 采用
                fake_AB = torch.cat((self.real_Ab, self.fake_b), 1) # 将两个图像进行cat在通道层进行叠加
            else:
                fake_AB = self.fake_b
            pred_fake = self.netD2(fake_AB)
            self.loss_G2_GAN = self.criterionGAN(pred_fake, True)  # 鉴别器计算BCE损失，True代表标签 自己手动打标签 loss = log(pred)
        else:
            self.loss_G2_GAN = 0

        # Second, G(A) = B
        self.loss_G2_L1 = self.criterionL1(self.fake_b, self.real_Bb)   # 生成器的L1损失
        # vgg loss 四个特征层的信息进行L1损失
        self.real_Bb_features = self.vgg(self.real_Bb)
        self.fake_Bb_features = self.vgg(self.fake_b)
        self.loss_vgg_G2 = self.criterionL1(self.fake_Bb_features[1], self.real_Bb_features[1]) * 1 + \
                        self.criterionL1(self.fake_Bb_features[2], self.real_Bb_features[2]) * 1 + \
                        self.criterionL1(self.fake_Bb_features[3],self.real_Bb_features[3]) * 1 + \
                        self.criterionL1(self.fake_Bb_features[0], self.real_Bb_features[0]) * 1
        # # TV loss 固定公式，只和生成的图片有关系，即fake_B
        # # torch.abs(self.fake_B[:, :, :, 1:] - self.fake_B[:, :, :, :-1]): 这个操作是计算上面两步获取的两个张量之间的元素级绝对差值。这样做可以得到相邻像素在垂直方向上的绝对差值。
        # # 计算的目的通常是用来作为一种正则化项,用于鼓励生成器网络生成平滑的图像。
        # diff_i = torch.sum(torch.abs(self.fake_b[:, :, :, 1:] - self.fake_b[:, :, :, :-1]))  # 列
        # diff_j = torch.sum(torch.abs(self.fake_b[:, :, 1:, :] - self.fake_b[:, :, :-1, :]))  # 行
        # self.tv_loss_G2 = (diff_i + diff_j) / (320 * 256)  # 因为此时文章提取到的图像是320×256的尺寸
        #
        # ssim-loss 设计ssim损失进行约束函数
        # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
        X = ((self.real_Bb + 1) / 2)
        Y = ((self.fake_b + 1) / 2)

        self.loss_ssim_G2 = 1 - ssim(X, Y, data_range=1, size_average=True)  # data_range=1 输入范围为 0-1 size_average=True 输入每个值的平均值 loss_ssim ：越低越真


        # 分割像素损失

        if self.l2 == 320:   # 此时没有b_III的分割损失
            self.loss_b_II = self.criterionL1(self.fake_b_II, self.real_Bab_II)  # b_II的分割损失
            self.loss_b = self.loss_b_II
        else:
            self.loss_b_II = self.criterionL1(self.fake_b_II, self.real_Bab_II)  # b_II的分割损失
            self.loss_b_III = self.criterionL1(self.fake_b_III, self.real_Bb_III)  # b_III的分割损失
            self.loss_b = (self.loss_b_II + self.loss_b_III) / 2

        # 步长损失
        if self.l1 == 0 or self.l2 == 320 or self.d_ls >= 0:  # 此时没有步长损失
            self.loss_b_step = 0
        else:
            self.loss_b_step = self.criterionL2(self.fake_b_step, self.real_Bb_step)



        # loss_G = loss_G_GAN × 0.03 + loss_G_L1 × 1 + loss_vgg × 1 + tv_loss × 1.25 + loss_ssim × 1.25
        # self.loss_G2 = self.loss_G2_GAN * self.w_gan + self.loss_G2_L1 + self.loss_vgg_G2 * self.w_vgg + self.tv_loss_G2 * self.w_tv + self.w_ss * self.loss_ssim_G2 + self.loss_b*self.w_ab
        # self.loss_G2 = self.loss_G2_GAN * self.w_gan + self.loss_vgg_G2 * self.w_vgg + self.tv_loss_G2 * self.w_tv + self.w_ss * self.loss_ssim_G2 + self.loss_b * self.w_ab + self.loss_b_step * self.w_step
        self.loss_G2 = self.loss_G2_GAN * self.w_gan + self.loss_vgg_G2 * self.w_vgg + self.w_ss * self.loss_ssim_G2 + self.loss_b * self.w_ab + self.loss_b_step * self.w_step

        self.loss_G2.backward() # 反向传播



    def Duibi(self,line):
        self.line = line
        # if self.loss_a_II > self.loss_b_II:
        #     self.line = self.line - 1
        #
        # elif self.loss_a_II < self.loss_b_II:
        #     self.line = self.line + 1
        # else:
        #     self.line = self.line
        PSNR_a, SSIM_a,PSNR_b,SSIM_b = self.return_PSNR_SSIM_a_b()
        lian_a = PSNR_a * 0.05 + SSIM_a - self.loss_a_II
        lian_b = PSNR_b * 0.05 + SSIM_b - self.loss_b_II
        if lian_a > lian_b:
            self.line = self.line + 1
        elif lian_a < lian_b:
            self.line = self.line - 1
        else:
            self.line = self.line
        return self.line


    # 为什么要先更新D然后再更新G，因为生成器生成的图像自己是分辨不出真假的，因此需要鉴别器分辨出图像真假然后得到图像的真假分布，因此才可以得出循环损失，才能更好的告诉G怎么进行梯度更新
    def optimize_parameters(self,l1,l2,line,best_ls_x,d_ls):
        self.forward(l1,l2,best_ls_x,d_ls)   # self.fake_B = self.netG(self.real_A)  得到生成器生成的图像
        # 更新 D1
        if self.use_gan == 1:
            self.set_requires_grad(self.netD1, True)  # 告诉鉴别器网络的需要梯度参数
            self.optimizer_D1.zero_grad()   # D优化器进行梯度清零
            self.backward_D1()              # 反向传播D网络数据
            self.optimizer_D1.step()        # 对数据进行更新
        else:
            self.loss_D1_fake = 0
            self.loss_D1_real = 0

        # 更新 G1
        self.set_requires_grad(self.netD1, False)   # 告诉鉴别器网络的不需要梯度参数 此时只更新生成器梯度
        self.optimizer_G1.zero_grad()
        self.backward_G1()
        self.optimizer_G1.step()

        # 更新D2

        if self.use_gan == 1:
            self.set_requires_grad(self.netD2, True)  # 告诉鉴别器网络的需要梯度参数
            self.optimizer_D2.zero_grad()   # D优化器进行梯度清零
            self.backward_D2()              # 反向传播D网络数据
            self.optimizer_D2.step()        # 对数据进行更新
        else:
            self.loss_D2_fake = 0
            self.loss_D2_real = 0

        # 更新 G2
        self.set_requires_grad(self.netD2, False)   # 告诉鉴别器网络的不需要梯度参数 此时只更新生成器梯度
        self.optimizer_G2.zero_grad()
        self.backward_G2()
        self.optimizer_G2.step()

        # 计算直线偏移量
        line = self.Duibi(line)


        return line

    # ---------同时判断两个生成器的相同损失结果------------------------



    def PSNR_SSIM(self, fake, real):   # 输入一个生成图片和一个真值图像进行计算生成图像的PSNR和ssim，要求输入为np

        # pnsr
        mse = np.mean((fake - real) ** 2)
        # psnr = 10*math.log10(1/mse)
        psnr = 10 * np.log10(255 * 255 / mse)

        # ssim
        ssim = compare_ssim(fake, real, win_size=3, channel_axis=2, data_range=255)
        # ssim = compare_ssim(a, b, win_size=11, channel_axis=2, data_range=255)

        return psnr, ssim

    def return_PSNR_SSIM(self):

        # 得到输出的tensor图像
        tensor_fake_a = self.fake_a
        tensor_fake_b = self.fake_b

        tensor_real_Ba =self.real_Ba
        tensor_real_Bb =self.real_Bb

        # 将输入的tensor图像转换为 HWC的图像格式
        img_fake_a = util.tensor2im(tensor_fake_a)
        img_fake_b = util.tensor2im(tensor_fake_b)
        img_real_Ba = util.tensor2im(tensor_real_Ba)
        img_real_Bb = util.tensor2im(tensor_real_Bb)


        # # 转换格式
        # img_fake_a = cv2.cvtColor(img_fake_a, cv2.COLOR_BGR2RGB)
        # img_fake_b = cv2.cvtColor(img_fake_b, cv2.COLOR_BGR2RGB)
        # img_real_Ba = cv2.cvtColor(img_real_Ba, cv2.COLOR_BGR2RGB)
        # img_real_Bb = cv2.cvtColor(img_real_Bb, cv2.COLOR_BGR2RGB)

        # 计算PSNR ,SSIM
        PSNR_a , SSIM_a = self.PSNR_SSIM(img_fake_a, img_real_Ba)
        PSNR_b, SSIM_b = self.PSNR_SSIM(img_fake_b, img_real_Bb)

        return PSNR_a, SSIM_a,PSNR_b,SSIM_b

    def return_PSNR_SSIM_a_b(self):

        # 得到输出的tensor图像
        tensor_fake_a = self.fake_a
        tensor_fake_b = self.fake_b

        tensor_real_Ba =self.real_Ba
        tensor_real_Bb =self.real_Bb

        # 将输入的tensor图像转换为 HWC的图像格式
        img_fake_a = util.tensor2im(tensor_fake_a)
        img_fake_b = util.tensor2im(tensor_fake_b)
        img_real_Ba = util.tensor2im(tensor_real_Ba)
        img_real_Bb = util.tensor2im(tensor_real_Bb)


        # # 转换格式
        # img_fake_a = cv2.cvtColor(img_fake_a, cv2.COLOR_BGR2RGB)
        # img_fake_b = cv2.cvtColor(img_fake_b, cv2.COLOR_BGR2RGB)
        # img_real_Ba = cv2.cvtColor(img_real_Ba, cv2.COLOR_BGR2RGB)
        # img_real_Bb = cv2.cvtColor(img_real_Bb, cv2.COLOR_BGR2RGB)

        # 计算PSNR ,SSIM
        PSNR_a , SSIM_a = self.PSNR_SSIM(img_fake_a, img_real_Ba)
        PSNR_b, SSIM_b = self.PSNR_SSIM(img_fake_b, img_real_Bb)

        return PSNR_a, SSIM_a,PSNR_b,SSIM_b











