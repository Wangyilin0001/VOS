import os.path
import random

import numpy
import torchvision.transforms as transforms
import torch
from .base_dataset import BaseDataset
from .image_folder import make_dataset
from PIL import Image

import cv2
import skimage
from skimage import data, exposure, img_as_float,io
import numpy as np
from util import util

class AlignedDataset(BaseDataset):  # 继承BaseDataset父类，实现将图像进行编号分类
    def one(self):
        # 初始化空数组
        AlignedDataset.l1_results = np.empty(0,dtype=int)
        AlignedDataset.l2_results = np.empty(0,dtype=int)   # 保存总的序列
        AlignedDataset.l1_results_3 = np.empty(0,dtype=int)  # 积攒的数值
        AlignedDataset.l2_results_3 = np.empty(0,dtype=int)

    def initialize(self, opt, line = 0, epoch =1,two = 1):
        self.opt = opt
        # self.root = opt.dataroot
        self.root = ''
        self.dir_AB = os.path.join(opt.dataroot, opt.phase) # 获得图像地址
        self.AB_paths = sorted(make_dataset(self.dir_AB)) # 返回图像的路径，读取的是一张图像
        assert(opt.resize_or_crop == 'resize_and_crop') # 检查命令是否是需要的
        self.two = two
        # --------------------调试-------------------------------------
        AlignedDataset.l1_results = np.empty(0, dtype=int)
        AlignedDataset.l2_results = np.empty(0, dtype=int)  # 保存总的序列
        AlignedDataset.l1_results_3 = [131]
        AlignedDataset.l2_results_3 = [239]
        AlignedDataset.line_l1 = 0
        AlignedDataset.line_l2 = 0
        AlignedDataset.d_ls = 0
        AlignedDataset.best_ls = 0
        AlignedDataset.best_ls_x = 0
        print('输入l1：',AlignedDataset.line_l1)
        print('输入l2：', AlignedDataset.line_l2)
        # ------------------------------------------------------------
        print('line',line)

        if two == 1 and len(AlignedDataset.l1_results_3)==0:

            if line > 0:    # 首先判断是否有偏移量,大于0向右移动
                # --------------计算最优滑动距离--------------------------------
                AlignedDataset.best_ls,AlignedDataset.best_ls_x = self.huadong_ls(line)
                # --------------计算最优滑动距离--------------------------------
                AlignedDataset.line_l1 = AlignedDataset.line_l1 + AlignedDataset.best_ls
                AlignedDataset.line_l2 = 108 + AlignedDataset.line_l1
                AlignedDataset.d_ls = 1
            elif line < 0:
                # --------------计算最优滑动距离--------------------------------
                AlignedDataset.best_ls,AlignedDataset.best_ls_x= self.huadong_ls(line)
                # --------------计算最优滑动距离--------------------------------
                AlignedDataset.line_l1 = AlignedDataset.line_l1 - AlignedDataset.best_ls
                AlignedDataset.line_l2 = 108 + AlignedDataset.line_l1
                AlignedDataset.d_ls = -1
            else:
                if epoch != 1:   # 如果是不是第一次输入则执行上次的累加结果
                    AlignedDataset.line_l1 = AlignedDataset.line_l1
                    AlignedDataset.line_l2 = 108 + AlignedDataset.line_l1
                    AlignedDataset.d_ls = 0
                    AlignedDataset.best_ls = 0
                    AlignedDataset.best_ls_x =0
                else:       # 如果是第一次输入则执行初始化
                    AlignedDataset.line_l1 = int(self.opt.loadSize_w / 3)
                    AlignedDataset.line_l2 = 108 + AlignedDataset.line_l1
                    AlignedDataset.d_ls = 0
                    AlignedDataset.best_ls = 0
                    AlignedDataset.best_ls_x = 0
            # 闸值限制
            if AlignedDataset.line_l1 < 0:
                AlignedDataset.line_l1 = 0
            elif AlignedDataset.line_l1 > 212:
                AlignedDataset.line_l1 = 212
            else:
                AlignedDataset.line_l1 = AlignedDataset.line_l1

            if AlignedDataset.line_l2 < 108:
                AlignedDataset.line_l2 = 108
            elif AlignedDataset.line_l2 > 320:
                AlignedDataset.line_l2 = 320
            else:
                AlignedDataset.line_l2 = AlignedDataset.line_l2

            AlignedDataset.l1_results = np.append(AlignedDataset.l1_results, AlignedDataset.line_l1)
            AlignedDataset.l2_results = np.append(AlignedDataset.l2_results, AlignedDataset.line_l2)
            if AlignedDataset.l1_results.size <= 1:
                AlignedDataset.l1_results_3 = AlignedDataset.l1_results.copy()  # 如果元素个数少于等于3，则赋值为results
                AlignedDataset.l2_results_3 = AlignedDataset.l2_results.copy()  # 如果元素个数少于等于3，则赋值为results
            else:
                AlignedDataset.l1_results_3 = AlignedDataset.l1_results[-3:]  # 如果元素个数大于3，则赋值为results的最新三个值
                AlignedDataset.l2_results_3 = AlignedDataset.l2_results[-3:]  # 如果元素个数大于3，则赋值为results的最新三个值

            self.line_l1 = AlignedDataset.l1_results_3[-1]
            self.line_l2 = AlignedDataset.l2_results_3[-1]


        elif two == 1 and len(AlignedDataset.l1_results_3):
            self.line_l1 = AlignedDataset.l1_results_3[-1]
            self.line_l2 = AlignedDataset.l2_results_3[-1]

        elif two == 2:
            print('self.l1_results_3', AlignedDataset.l1_results_3)
            print('self.l2_results_3', AlignedDataset.l2_results_3)
            self.line_l1 = AlignedDataset.l1_results_3[-1]
            self.line_l2 = AlignedDataset.l2_results_3[-1]
            AlignedDataset.l1_results_3 = np.delete(AlignedDataset.l1_results_3, -1)
            AlignedDataset.l2_results_3 = np.delete(AlignedDataset.l2_results_3, -1)
            AlignedDataset.d_ls = AlignedDataset.d_ls
            AlignedDataset.best_ls = AlignedDataset.best_ls
            AlignedDataset.best_ls_x = AlignedDataset.best_ls_x

        print('self.l1_results', AlignedDataset.l1_results)
        print('self.l2_results', AlignedDataset.l2_results)

        print('self.l1_results_3',AlignedDataset.l1_results_3)
        print('self.l2_results_3', AlignedDataset.l1_results_3)

        print('输出l1：', self.line_l1)
        print('输出l2：', self.line_l2)

        return self.line_l1, self.line_l2, AlignedDataset.best_ls_x, AlignedDataset.d_ls

    def __getitem__(self, index):


        AB_path = self.AB_paths[index]    # 读取图片

        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize_w, self.opt.loadSize_h), Image.BICUBIC)
        B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize_w, self.opt.loadSize_h), Image.BICUBIC)

        # 裁剪A得到a,b
        A_w,A_h = A.size
        # l1 = int((A_w+1)/3)
        # l2 = int(A_w -l1)


        l1 = self.line_l1
        l2 = self.line_l2


        a = A.crop((0, 0, l2, A_h)).resize((256, 256),Image.BICUBIC)
        b = A.crop((l1, 0, A_w, A_h),).resize((256, 256),Image.BICUBIC)
        Ba = B.crop((0, 0, l2, A_h)).resize((256, 256),Image.BICUBIC)
        Bb = B.crop((l1, 0, A_w, A_h)).resize((256, 256),Image.BICUBIC)
        Aa = A.crop((0, 0, l2, A_h)).resize((256, 256),Image.BICUBIC)          # 红外A中的a
        Ab = A.crop((l1, 0, A_w, A_h)).resize((256, 256),Image.BICUBIC)        # 红外A中的b


        # 将 a,b,Aa，Ba转换为array类型
        a = np.array(a)
        b = np.array(b)
        Ba = np.array(Ba)
        Bb = np.array(Bb)
        A = np.array(A)
        B = np.array(B)
        Aa = np.array(Aa)
        Ab = np.array(Ab)



        a = Image.fromarray(np.uint8(a)).convert('RGB')  # 原始的 NumPy 数组 A 被转换为 PIL 的 Image 对象,并且保证了数据类型和色彩空间的正确性。
        b = Image.fromarray(np.uint8(b)).convert('RGB')
        Ba = Image.fromarray(np.uint8(Ba)).convert('RGB')  # 原始的 NumPy 数组 A 被转换为 PIL 的 Image 对象,并且保证了数据类型和色彩空间的正确性。
        Bb = Image.fromarray(np.uint8(Bb)).convert('RGB')
        A = Image.fromarray(np.uint8(A)).convert('RGB')  # 原始的 NumPy 数组 A 被转换为 PIL 的 Image 对象,并且保证了数据类型和色彩空间的正确性。
        B = Image.fromarray(np.uint8(B)).convert('RGB')
        Aa = Image.fromarray(np.uint8(Aa)).convert('RGB')  # 原始的 NumPy 数组 A 被转换为 PIL 的 Image 对象,并且保证了数据类型和色彩空间的正确性。
        Ab = Image.fromarray(np.uint8(Ab)).convert('RGB')



        a = transforms.ToTensor()(a)  # (0 - 255) --> (0 - 1)
        b = transforms.ToTensor()(b)
        Ba = transforms.ToTensor()(Ba)  # (0 - 255) --> (0 - 1)
        Bb = transforms.ToTensor()(Bb)
        A = transforms.ToTensor()(A)  # (0 - 255) --> (0 - 1)
        B = transforms.ToTensor()(B)
        Aa = transforms.ToTensor()(Aa)  # (0 - 255) --> (0 - 1)
        Ab = transforms.ToTensor()(Ab)


        # img_a =  a.unsqueeze(0)  # 给矢量数组增添一个维度
        # img_b =  b.unsqueeze(0)  # 给矢量数组增添一个维度
        # img_a = util.tensor2im(img_a)
        # img_b = util.tensor2im(img_b)
        #
        # cv2.imshow('1', img_a)  # 生成的a
        # cv2.imshow('2', img_b)  # 生成的b
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)
        Ba = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(Ba)
        Bb = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(Bb)
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
        Aa = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(Aa)
        Ab = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(Ab)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc


        if input_nc == 1:  # RGB to gray
            tmp_a = a[0, ...] * 0.299 + a[1, ...] * 0.587 + a[2, ...] * 0.114
            a = tmp_a.unsqueeze(0)
            tmp_b = b[0, ...] * 0.299 + b[1, ...] * 0.587 + b[2, ...] * 0.114
            b = tmp_b.unsqueeze(0)
            tmp_A = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp_A.unsqueeze(0)
            tmp_Aa = Aa[0, ...] * 0.299 + Aa[1, ...] * 0.587 + Aa[2, ...] * 0.114
            Aa = tmp_Aa.unsqueeze(0)
            tmp_Ab = Ab[0, ...] * 0.299 + Ab[1, ...] * 0.587 + Ab[2, ...] * 0.114
            Ab = tmp_Ab.unsqueeze(0)



        if output_nc == 1:  # RGB to gray
            tmp_Ba = Ba[0, ...] * 0.299 + Ba[1, ...] * 0.587 + Ba[2, ...] * 0.114
            Ba = tmp_Ba.unsqueeze(0)
            tmp_Bb = Bb[0, ...] * 0.299 + Bb[1, ...] * 0.587 + Bb[2, ...] * 0.114
            Bb = tmp_Bb.unsqueeze(0)
            tmp_B = Bb[0, ...] * 0.299 + Bb[1, ...] * 0.587 + Bb[2, ...] * 0.114
            B = tmp_B.unsqueeze(0)


        # print('a.shape:',a.shape)
        # print('b.shape:',b.shape)
        # print('Ba.shape:',Ba.shape)
        # print('Bb.shape:',Bb.shape)
        # print('A.shape:',A.shape)
        # print('B.shape:',B.shape)
        # print('Aa.shape:', Aa.shape)
        # print('Ab.shape:', Ab.shape)

        return {'a': a, 'b': b,'Ba': Ba, 'Bb': Bb,'A':A,'B':B,'Aa':Aa,'Ab':Ab,
                'a_paths': AB_path, 'b_paths': AB_path,'Ba_paths': AB_path, 'Bb_paths': AB_path,'A_paths': AB_path, 'B_paths': AB_path}




    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'

    def huadong_ls(self,line,ls_min=0, ls_max=10, x_min=-5, x_max=5, channel=256):
        # 初始化最小绝对值和对应的ls, x
        Xl_l1 = self.line_l1
        Xl_l2 = self.line_l2
        min_e_abs = float('inf')
        best_ls = None
        best_x = None
        # 遍历可能的ls和x的取值
        if line != 0:
            for ls in range(ls_min, ls_max):  # 假设ls的范围
                for x in range(x_min, x_max):  # 假设x的范围
                    # for Xl in Xl_range:
                    # 计算Y和e
                    if line > 0 :
                        Y = ls * (ls + Xl_l2) / channel
                        e = ls + x - Y
                    elif line < 0 :
                        Y = ls * (320 + ls - Xl_l1) / channel
                        e = ls + x - Y
                    elif ls + x <= 0:
                        e =0

                    # 检查当前e的绝对值是否为最小
                    if abs(e) < min_e_abs and e != 0:
                        min_e_abs = abs(e)
                        best_ls = ls
                        best_x = x
        else:
            best_ls = 0
            best_x = 0
        # 打印结果
        best_ls_x = best_ls + best_x
        print(f"最小绝对值的e: {min_e_abs}, 对应的ls: {best_ls}, x: {best_x},ls_x:{best_ls_x}")
        return best_ls,best_ls_x
