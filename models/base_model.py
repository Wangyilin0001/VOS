import csv
import os
import torch
from collections import OrderedDict
from . import networks
from torchstat import stat


class BaseModel():

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser
    
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []

    def set_input(self, input):
        self.input = input

    def forward(self,l1,l2,best_ls_x,d_ls):
        pass

    # load and print networks; create schedulers
    def setup(self, opt, parser=None):
        if self.isTrain:
            # 段代码是在为每个优化器创建一个对应的学习率调度器,并将它们收集到一个列表中。这些调度器可以在训练过程中动态调整学习率,从而提高模型的训练效果。动态更新学习率
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    # make models eval mode during test time
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self,l1,l2,best_ls_x,d_ls):
        with torch.no_grad():
            self.forward(l1,l2,best_ls_x,d_ls)

    def compute_visuals(self):   # 添加新的可视化设计，可以将注意力图放在该模块中
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self, l1, l2, line,best_ls_x,d_ls):
        return line

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # return visualization images. train.py will display these images, and save the images to a html
    # 这个函数/方法需要返回一些可视化图像,这些图像将在 train.py 脚本中显示,并保存到一个 HTML 文件中
    def get_current_visuals(self):
        visual_ret = OrderedDict()#按照有序插入顺序存储
        for name in self.visual_names:
            if isinstance(name, str):   # 对于每个 name，检查它是否是一个字符串类型。
                visual_ret[name] = getattr(self, name)  # 如果 name 是一个字符串,那么使用 getattr(self, name) 获取 self 对象中名为 name 的属性值。
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    # 返回培训损失/错误。train.py会将这些错误作为调试信息打印出来
    def get_current_losses(self):
        errors_ret = OrderedDict()#按照有序插入顺序存储
        for name in self.loss_names:
            if isinstance(name, str):          #isinstance()用来判断一个对象是否是一个已知的类型
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # save models to the disk
    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)  # 保存的为pickle 通用模型

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                print(net)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print(net)
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                print(state_dict.keys())
                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)  # getattr 是 Python 中的一个内置函数,用于获取对象的属性值。
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                f2 = open(r'D:\GAN-NET\CDGAN\number.csv', 'a', newline='')
                csv_writer = csv.writer(f2)
                csv_writer.writerow([name,num_params / 1e6])
                f2.close()
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))  # G D
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):  # 检测是否是一个列表
            nets = [nets]
        for net in nets:
            if net is not None:   # 如果net不为空
                for param in net.parameters():  # 遍历每个神经网络模型的所有参数。
                    param.requires_grad = requires_grad   # 将每个参数的 requires_grad 属性设置为输入的 requires_grad 值。
