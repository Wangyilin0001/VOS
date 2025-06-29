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




class VOSModel(BaseModel):
    def name(self):
        return 'VOSModel'

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
        self.visual_names = ['real_Aa','fake_a', 'real_Ba','real_Ab','fake_b','real_Bb','real_A','real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G1', 'G2', 'D1','D2']
        else:  # during test time, only load Gs
            self.model_names = ['G1','G2']
        # use_gan
        self.use_gan = opt.use_GAN
        self.w_vgg = opt.w_vgg  # The proportion of vgg networks: 1
        self.w_tv = opt.w_tv    # The proportion of tv loss: 1.25
        self.w_gan = opt.w_gan  # The loss ratio of gan networks: 0.03
        self.w_ss = opt.w_ss    # The proportion of ssim loss: 1.25
        self.w_ab = opt.w_ab    # The proportion of pixel loss in segmentation: 1
        self.w_step = opt.w_step # The proportion of step size loss: 1
        self.use_condition = opt.use_condition  #  1

        self.netG1 = networks.define_G1(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG1, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        self.netG2 = networks.define_G2(opt.input_nc, opt.output_nc, opt.ngf,
                                        opt.which_model_netG2, opt.norm, not opt.no_dropout, opt.init_type,
                                        self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if self.use_condition == 1:

                self.netD1 = networks.define_D1(opt.input_nc + opt.output_nc, opt.ndf,
                                              opt.which_model_netD1,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
                self.netD2 = networks.define_D2(opt.input_nc + opt.output_nc, opt.ndf,
                                                opt.which_model_netD2,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            else:

                self.netD1 = networks.define_D1(opt.input_nc, opt.ndf,
                                              opt.which_model_netD1,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
                self.netD2 = networks.define_D1(opt.input_nc, opt.ndf,
                                                opt.which_model_netD2,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)


        if self.isTrain:

            self.fake_Aa_pool = ImagePool(opt.pool_size)
            self.fake_Ab_pool = ImagePool(opt.pool_size)
            # define loss functions
            if opt.which_model_netD1 == 'multi':
                self.criterionGAN = networks.GANLoss_multi(use_lsgan=not opt.no_lsgan).to(self.device)
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            # load vgg network
            self.vgg = Vgg16().type(torch.cuda.FloatTensor)

            # initialize optimizers
            self.optimizers = []

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

        self.real_a = input['a' if AtoB else 'b'].to(self.device)
        self.real_b = input['b' if AtoB else 'a'].to(self.device)
        self.real_Ba = input['Ba' if AtoB else 'Bb'].to(self.device)
        self.real_Bb = input['Bb' if AtoB else 'Ba'].to(self.device)
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_Aa = input['Aa' if AtoB else 'Ab'].to(self.device)
        self.real_Ab = input['Ab' if AtoB else 'Aa'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self, l1, l2,best_ls_x,d_ls):
        self.fake_a = self.netG1(self.real_a)
        self.fake_b = self.netG2(self.real_b)
        self.d_ls = d_ls
        self.l1 = l1
        self.l2 = l2

        # print('test:l1',l1)
        # print('test:l2', l2)
        p1 = 256 / l2
        p2 = 256 / (320 - l1)
        l1_p1 = l1 * p1
        l2_p2 = (l2 - l1) * p2

        ll1 = math.ceil(l1_p1)   # Round upwards
        ll2 = int(l2_p2)         # Round down
        # print('Input deformation:l1-1', ll1)
        # print('Input deformation:l2-2', ll2)
        self.ll1 = ll1
        self.ll2 = ll2
        #----------------------Visualization----------------------------

        # img_fake_a = util.tensor2im(self.fake_a)
        # img_fake_b = util.tensor2im(self.fake_b)
        # img_real_a = util.tensor2im(self.real_a)
        # img_real_b = util.tensor2im(self.real_b)

        # cv2.imshow('1', img_fake_a)
        # cv2.imshow('2', img_fake_b)
        # cv2.imshow('3', img_real_a)
        # cv2.imshow('4', img_real_b)
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # ----------------------Visualization----------------------------

        # Segmentation and calculation: fake_a_I,fake_a_II,fake_b_I,fake_b_II
        # Cut

        self.fake_a_I = self.fake_a[0:1,0:3,0:256,0:ll1]
        #print('self.fake_a_I.shape:',self.fake_a_I.shape)
        #print('self.fake_a.shape:', self.fake_a.shape)
        self.fake_a_II = self.fake_a[0:1,0:3,0:256,ll1:256]
        #print('self.fake_a_II.shape:',self.fake_a_II.shape)
        self.fake_b_II = self.fake_b[0:1,0:3,0:256,0:ll2]
        #print('self.fake_b_II.shape:',self.fake_b_II.shape)
        self.fake_b_III = self.fake_b[0:1,0:3,0:256,ll2:256]
        #print('self.fake_b_III.shape:',self.fake_b_III.shape)
        #print('self.real_Ba.shape:', self.real_Ba.shape)
        self.real_Ba_I = self.real_Ba[0:1,0:3,0:256,0:ll1]
        # print('self.real_Ba_I.shape:', self.real_Ba_I.shape)
        self.real_Ba_II = self.real_Ba[0:1,0:3,0:256,ll1:256]
        self.real_Bb_II = self.real_Ba[0:1, 0:3, 0:256, 0:ll2]
        # print('self.real_Bab_II.shape:', self.real_Bab_II.shape)
        self.real_Bb_III = self.real_Bb[0:1,0:3,0:256,ll2:256]
        # print('self.real_Bb_III.shape:', self.real_Bb_III.shape)
        # 步长损失
        self.fake_a_step = self.fake_a[0:1,0:3,0:256,256-best_ls_x:256]
        self.fake_b_step = self.fake_b[0:1, 0:3, 0:256, 0:best_ls_x]
        self.real_Ba_step = self.real_Ba[0:1,0:3,0:256,256-best_ls_x:256]
        self.real_Bb_step = self.real_Bb[0:1, 0:3, 0:256, 0:best_ls_x]
        # ------------------Visualization---------------------------------

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

        # ------------------Visualization---------------------------------

        #  Solve the image of the intersecting area
        # self.fake_ab_II = (self.fake_a_II+self.fake_b_II)/2
        #
        # self.l1 = l1
        # self.l2 = l2
        #
        #
        # if l1 == 0:
        #     self.fake_ab_II_np = self.fake_ab_II[0].cpu().detach().numpy()
        #     self.fake_b_III_np = self.fake_b_III[0].cpu().detach().numpy()
        #
        #     self.fake_ab_II_np = np.transpose(self.fake_ab_II_np, (1, 2, 0))
        #     self.fake_b_III_np = np.transpose(self.fake_b_III_np, (1, 2, 0))
        #
        #     self.fake_ab_II_np = cv2.resize(self.fake_ab_II_np, (l2-l1, 256))
        #     self.fake_b_III_np = cv2.resize(self.fake_b_III_np, (320-l2, 256))
        #
        #     self.fake_AB = np.concatenate((self.fake_ab_II_np,self.fake_b_III_np), axis=1)
        #
        # elif l2 == 320:
        #     self.fake_a_I_np = self.fake_a_I[0].cpu().detach().numpy()
        #     self.fake_ab_II_np = self.fake_ab_II[0].cpu().detach().numpy()
        #
        #     self.fake_a_I_np = np.transpose(self.fake_a_I_np, (1, 2, 0))
        #     self.fake_ab_II_np = np.transpose(self.fake_ab_II_np, (1, 2, 0))
        #
        #     self.fake_a_I_np = cv2.resize(self.fake_a_I_np, (l1, 256))
        #     self.fake_ab_II_np = cv2.resize(self.fake_ab_II_np, (l2-l1, 256))
        #
        #     self.fake_AB = np.concatenate((self.fake_a_I_np, self.fake_ab_II_np), axis=1)
        #
        # else:
        #     self.fake_a_I_np = self.fake_a_I[0].cpu().detach().numpy()
        #     self.fake_ab_II_np = self.fake_ab_II[0].cpu().detach().numpy()
        #     self.fake_b_III_np = self.fake_b_III[0].cpu().detach().numpy()
        #
        #     self.fake_a_I_np = np.transpose(self.fake_a_I_np, (1, 2, 0))
        #     self.fake_ab_II_np = np.transpose(self.fake_ab_II_np, (1, 2, 0))
        #     self.fake_b_III_np = np.transpose(self.fake_b_III_np, (1, 2, 0))
        #
        #     self.fake_a_I_np = cv2.resize(self.fake_a_I_np, (l1, 256))
        #     self.fake_ab_II_np = cv2.resize(self.fake_ab_II_np, (l2-l1, 256))
        #     self.fake_b_III_np = cv2.resize(self.fake_b_III_np, (320-l2, 256))
        #
        #     self.fake_AB = np.concatenate((self.fake_a_I_np, self.fake_ab_II_np, self.fake_b_III_np),axis=1)
        #
        #
        # self.fake_AB = transforms.ToTensor()(self.fake_AB).to(self.device)
        # self.fake_AB = self.fake_AB.unsqueeze(0)



    def backward_D1(self):
        # Fake
        # stop backprop to the generator by detaching fake_B

        if self.use_condition == 1:
            fake_AB = self.fake_Aa_pool.query(torch.cat((self.real_Aa, self.fake_a), 1))
        else:
            fake_AB = self.fake_a
        pred_fake = self.netD1(fake_AB.detach())
        # loss = -(1 - y) * log(1 - pred) - y * log(pred)  格式为（pred，T/F）
        self.loss_D1_fake = self.criterionGAN(pred_fake, False)

        # Real
        if self.use_condition == 1:
            real_AB = torch.cat((self.real_Aa, self.real_Ba), 1)
        else:
            real_AB = self.real_Ba
        pred_real = self.netD1(real_AB)
        self.loss_D1_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D1 = (self.loss_D1_fake + self.loss_D1_real) * 0.5

        self.loss_D1.backward()

    def backward_G1(self):
        # First, G(A) should fake the discriminator
        if self.use_gan == 1:
            if self.use_condition == 1:
                fake_AB = torch.cat((self.real_Aa, self.fake_a), 1)
            else:
                fake_AB = self.fake_a
            pred_fake = self.netD1(fake_AB)
            self.loss_G1_GAN = self.criterionGAN(pred_fake, True)
        else:
            self.loss_G1_GAN = 0

        # Second, G(A) = B
        # self.loss_G1_L1 = self.criterionL1(self.fake_a, self.real_Ba)

        self.real_Ba_features = self.vgg(self.real_Ba)
        self.fake_Ba_features = self.vgg(self.fake_a)
        self.loss_vgg_G1 = self.criterionL1(self.fake_Ba_features[1], self.real_Ba_features[1]) * 1 + \
                        self.criterionL1(self.fake_Ba_features[2], self.real_Ba_features[2]) * 1 + \
                        self.criterionL1(self.fake_Ba_features[3],self.real_Ba_features[3]) * 1 + \
                        self.criterionL1(self.fake_Ba_features[0], self.real_Ba_features[0]) * 1

        # ssim-loss
        # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
        X = ((self.real_Ba + 1) / 2)
        Y = ((self.fake_a + 1) / 2)
        #
        self.loss_ssim_G1 = 1 - ssim(X, Y, data_range=1, size_average=True)

        # split pixel loss

        if self.l1 ==0:   # At this time, there is no segmentation loss of b_I
            self.loss_a_II = self.criterionL1(self.fake_a_II, self.real_Ba_II)   # The split loss of a_II
            self.loss_a = self.loss_a_II
        else:
            self.loss_a_I = self.criterionL1(self.fake_a_I, self.real_Ba_I)  # The split loss of a_I
            self.loss_a_II = self.criterionL1(self.fake_a_II, self.real_Ba_II)  # The split loss of a_II
            self.loss_a = self.loss_a_I + self.loss_a_II

        # step loss

        if self.l1 == 0 or self.l2 == 320 or self.d_ls<=0:  # There is no step size loss at this time
            self.loss_a_step = 0
        else:
            self.loss_a_step = self.criterionL2(self.fake_a_step, self.real_Ba_step)

        self.loss_G1 = self.loss_G1_GAN * self.w_gan + self.loss_vgg_G1 * self.w_vgg + self.w_ss * self.loss_ssim_G1 + self.loss_a * self.w_ab + self.loss_a_step * self.w_step

        self.loss_G1.backward()

    def backward_D2(self):
        if self.use_condition == 1:
            fake_AB = self.fake_Ab_pool.query(torch.cat((self.real_Ab, self.fake_b), 1))
        else:
            fake_AB = self.fake_b
        pred_fake = self.netD2(fake_AB.detach())
        # loss = -(1 - y) * log(1 - pred) - y * log(pred)  格式为（pred，T/F）
        self.loss_D2_fake = self.criterionGAN(pred_fake, False)

        # Real
        if self.use_condition == 1:
            real_AB = torch.cat((self.real_Ab, self.real_Bb), 1)
        else:
            real_AB = self.real_Bb
        pred_real = self.netD2(real_AB)
        self.loss_D2_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5

        self.loss_D2.backward()

    def backward_G2(self):
        # First, G(A) should fake the discriminator
        if self.use_gan == 1:
            if self.use_condition == 1:  # 采用
                fake_AB = torch.cat((self.real_Ab, self.fake_b), 1)
            else:
                fake_AB = self.fake_b
            pred_fake = self.netD2(fake_AB)
            self.loss_G2_GAN = self.criterionGAN(pred_fake, True)
        else:
            self.loss_G2_GAN = 0

        # Second, G(A) = B
        self.loss_G2_L1 = self.criterionL1(self.fake_b, self.real_Bb)

        self.real_Bb_features = self.vgg(self.real_Bb)
        self.fake_Bb_features = self.vgg(self.fake_b)
        self.loss_vgg_G2 = self.criterionL1(self.fake_Bb_features[1], self.real_Bb_features[1]) * 1 + \
                        self.criterionL1(self.fake_Bb_features[2], self.real_Bb_features[2]) * 1 + \
                        self.criterionL1(self.fake_Bb_features[3],self.real_Bb_features[3]) * 1 + \
                        self.criterionL1(self.fake_Bb_features[0], self.real_Bb_features[0]) * 1

        # ssim-loss
        # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
        X = ((self.real_Bb + 1) / 2)
        Y = ((self.fake_b + 1) / 2)

        self.loss_ssim_G2 = 1 - ssim(X, Y, data_range=1, size_average=True)


        if self.l2 == 320:
            self.loss_b_II = self.criterionL1(self.fake_b_II, self.real_Bb_II)
            self.loss_b = self.loss_b_II
        else:
            self.loss_b_II = self.criterionL1(self.fake_b_II, self.real_Bb_II)
            self.loss_b_III = self.criterionL1(self.fake_b_III, self.real_Bb_III)
            self.loss_b = (self.loss_b_II + self.loss_b_III) / 2

        if self.l1 == 0 or self.l2 == 320 or self.d_ls >= 0:
            self.loss_b_step = 0
        else:
            self.loss_b_step = self.criterionL2(self.fake_b_step, self.real_Bb_step)

        self.loss_G2 = self.loss_G2_GAN * self.w_gan + self.loss_vgg_G2 * self.w_vgg + self.w_ss * self.loss_ssim_G2 + self.loss_b * self.w_ab + self.loss_b_step * self.w_step

        self.loss_G2.backward()



    def OAS(self,line):
        self.line = line
        lian_a = self.loss_a_II/self.ll1
        lian_b = self.loss_b_II/self.ll2
        if lian_a > lian_b:
            self.line = self.line - 1
        elif lian_a < lian_b:
            self.line = self.line + 1
        else:
            self.line = self.line
        return self.line



    def optimize_parameters(self,l1,l2,line,best_ls_x,d_ls):
        self.forward(l1,l2,best_ls_x,d_ls)   # self.fake_B = self.netG(self.real_A)
        if self.use_gan == 1:
            self.set_requires_grad(self.netD1, True)
            self.optimizer_D1.zero_grad()
            self.backward_D1()
            self.optimizer_D1.step()
        else:
            self.loss_D1_fake = 0
            self.loss_D1_real = 0

        # G1
        self.set_requires_grad(self.netD1, False)
        self.optimizer_G1.zero_grad()
        self.backward_G1()
        self.optimizer_G1.step()

        # D2

        if self.use_gan == 1:
            self.set_requires_grad(self.netD2, True)
            self.optimizer_D2.zero_grad()
            self.backward_D2()
            self.optimizer_D2.step()
        else:
            self.loss_D2_fake = 0
            self.loss_D2_real = 0

        # G2
        self.set_requires_grad(self.netD2, False)
        self.optimizer_G2.zero_grad()
        self.backward_G2()
        self.optimizer_G2.step()

        # Calculate the straight-line offset
        line = self.OAS(line)


        return line


    def PSNR_SSIM(self, fake, real):

        # pnsr
        mse = np.mean((fake - real) ** 2)
        # psnr = 10*math.log10(1/mse)
        psnr = 10 * np.log10(255 * 255 / mse)

        # ssim
        ssim = compare_ssim(fake, real, win_size=3, channel_axis=2, data_range=255)
        # ssim = compare_ssim(a, b, win_size=11, channel_axis=2, data_range=255)

        return psnr, ssim

    def return_PSNR_SSIM(self):

        tensor_fake_a = self.fake_a
        tensor_fake_b = self.fake_b

        tensor_real_Ba =self.real_Ba
        tensor_real_Bb =self.real_Bb

        img_fake_a = util.tensor2im(tensor_fake_a)
        img_fake_b = util.tensor2im(tensor_fake_b)
        img_real_Ba = util.tensor2im(tensor_real_Ba)
        img_real_Bb = util.tensor2im(tensor_real_Bb)

        PSNR_a , SSIM_a = self.PSNR_SSIM(img_fake_a, img_real_Ba)
        PSNR_b, SSIM_b = self.PSNR_SSIM(img_fake_b, img_real_Bb)

        return PSNR_a, SSIM_a,PSNR_b,SSIM_b

    def return_PSNR_SSIM_a_b(self):

        tensor_fake_a = self.fake_a
        tensor_fake_b = self.fake_b

        tensor_real_Ba =self.real_Ba
        tensor_real_Bb =self.real_Bb

        img_fake_a = util.tensor2im(tensor_fake_a)
        img_fake_b = util.tensor2im(tensor_fake_b)
        img_real_Ba = util.tensor2im(tensor_real_Ba)
        img_real_Bb = util.tensor2im(tensor_real_Bb)


        # img_fake_a = cv2.cvtColor(img_fake_a, cv2.COLOR_BGR2RGB)
        # img_fake_b = cv2.cvtColor(img_fake_b, cv2.COLOR_BGR2RGB)
        # img_real_Ba = cv2.cvtColor(img_real_Ba, cv2.COLOR_BGR2RGB)
        # img_real_Bb = cv2.cvtColor(img_real_Bb, cv2.COLOR_BGR2RGB)

        PSNR_a , SSIM_a = self.PSNR_SSIM(img_fake_a, img_real_Ba)
        PSNR_b, SSIM_b = self.PSNR_SSIM(img_fake_b, img_real_Bb)

        return PSNR_a, SSIM_a,PSNR_b,SSIM_b











