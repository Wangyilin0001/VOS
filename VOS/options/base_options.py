import argparse
import os
from util import util
import torch
import models


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        #parser.add_argument('--dataroot',type=str, default=r'D:\GAN-NET\dataset\kaist_wash_day', help='path to images (should have subfolders train, val, etc)')
        parser.add_argument('--dataroot', type=str, default=r'E:\GAN-NET\dataset\test_FLIR_one',help='path to images (should have subfolders train, val, etc)')
        # parser.add_argument('--dataroot', type=str, default=r'E:\GAN-NET\dataset\test_FLIR',help='path to images (should have subfolders train, val, etc)')
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--loadSize_w', type=int, default=320, help='scale images to this size')
        parser.add_argument('--fineSize_w', type=int, default=320, help='then crop to this size')
        parser.add_argument('--loadSize_h', type=int, default=256, help='scale images to this size')
        parser.add_argument('--fineSize_h', type=int, default=256, help='then crop to this size')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--which_model_netG1', type=str, default='LK_GAN_G1', help='selects model to use for netG,pix2pix,MSUNet_G1,LK_GAN_G1')
        parser.add_argument('--which_model_netG2', type=str, default='LK_GAN_G2', help='selects model to use for netG,pix2pix,MSUNet_G2,LK_GAN_G2')
        parser.add_argument('--which_model_netD1', type=str, default='LK_GAN_D1', help='selects model to use for netD,basic_D1,LK_GAN_D1')
        parser.add_argument('--which_model_netD2', type=str, default='LK_GAN_D2', help='selects model to use for netD,basic_D2,LK_GAN_D2')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--dataset_mode', type=str, default='aligned',
                            help='chooses how datasets are loaded. [unaligned | aligned | single]')
        parser.add_argument('--model', type=str, default='CDGAN',
                            help='chooses which model to use. cycle_gan, pix2pix, test，CDGAN')   # ----交叉差异对抗网络
        parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--nThreads', default=6, type=int, help='# threads for loading data')   # 输入图像的数量，此时修改为6加快模型运行速度
        #parser.add_argument('--checkpoints_dir', type=str, default=r'D:\GAN-NET\MUGAN-me\MUGAN/checkpoints/kaist/', help='models are saved here')
        parser.add_argument('--checkpoints_dir', type=str, default=r'./checkpoint/', help='models are saved here')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization')
        parser.add_argument('--serial_batches', action='store_true', # action 执行，不为空定义为真
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost",
                            help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='Visdom 显示环境名称（默认为“main”）')
        parser.add_argument('--display_port', type=int, default=6070, help='visdom port of the web display')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                            help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--verbose', action='store_true',default=False, help='if specified, print more debugging information') # 打印模型信息  False or Ture
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')
        parser.add_argument('--use_GAN', default=1, help='1 is use gan')
        parser.add_argument('--w_gan', default=0.03, help='weight of the gan loss')  # gan损失系数吗，通用0.03
        parser.add_argument('--w_vgg', default=1, help='weight of the vgg loss')     # vgg损失系数吗，通用1
        parser.add_argument('--w_tv', default=1.25, help='weight of the tv loss')    # tv损失系数吗，通用1.25
        parser.add_argument('--w_ss', default=1.25, help='weight of the ms-ssim loss')  # ssim损失系数吗，通用1.25
        parser.add_argument('--w_ab', default=1, help='weight of the ab loss')  # 分割像素损失损失系数吗，通用1
        parser.add_argument('--w_step', default=1, help='weight of the step loss')  # 滑动损失系数吗，通用10
        parser.add_argument('--use_wandb', action='store_true', default=False, help='use wandb')

        parser.add_argument('--use_condition', default=1, help='1 means add condition in discriminator')


        # 自己修改参数
        parser.add_argument('--train_number', default=5, help='每几轮保存一下模型')
        parser.add_argument('--train_path', default=r'.\\result\train', help='训练结果保存地址')
        parser.add_argument('--test_path', default=r'.\\result\test', help='测试结果保存地址')



        self.initialized = True
        return parser

    def gather_options(self):

        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        # POSSIBLE FEATURE:
        # modify dataset-related parser options

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
