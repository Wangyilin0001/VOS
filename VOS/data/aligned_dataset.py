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
import csv

class AlignedDataset(BaseDataset):
    def one(self):

        AlignedDataset.l1_results = np.empty(0,dtype=int)
        AlignedDataset.l2_results = np.empty(0,dtype=int)
        AlignedDataset.l1_results_3 = np.empty(0,dtype=int)
        AlignedDataset.l2_results_3 = np.empty(0,dtype=int)

    def initialize(self, opt, line = 0, epoch =1,two = 1):
        self.opt = opt
        # self.root = opt.dataroot
        self.root = ''
        self.dir_AB = os.path.join(opt.dataroot, opt.phase) # Obtain the image address
        self.AB_paths = sorted(make_dataset(self.dir_AB)) # Return the path of the image, and what is read is an image
        assert(opt.resize_or_crop == 'resize_and_crop') # Check whether the command is necessary
        self.two = two
        AlignedDataset.change = 1

        # --------------------Test-------------------------------------
        '''
        When testing, the following lines of code need to be used to adjust the split lines L1 and L2. 
        During training, shielding is required.
        AlignedDataset.l1_results_3 = [x]
        AlignedDataset.l2_results_3 = [y]
        Where, x represents the pixel position of line L1, and y represents the pixel position of line L2.
        '''
        # AlignedDataset.l1_results = np.empty(0, dtype=int)
        # AlignedDataset.l2_results = np.empty(0, dtype=int)
        # AlignedDataset.l1_results_3 = [212]
        # AlignedDataset.l2_results_3 = [320]
        # AlignedDataset.line_l1 = 0
        # AlignedDataset.line_l2 = 0
        # AlignedDataset.d_ls = 0
        # AlignedDataset.best_ls = 0
        # AlignedDataset.best_ls_x = 0
        # print('test_l1：',AlignedDataset.line_l1)
        # print('test_l2：', AlignedDataset.line_l2)

        # ------------------------------------------------------------
        print('line',line)
        f5 = open(r'./line.csv', 'a', newline='')
        csv_writer = csv.writer(f5)
        csv_writer.writerow(['epoch:%d,' % epoch +'line:%d,' % line])
        f5.close()

        if two == 1 and len(AlignedDataset.l1_results_3)==0:

            if line > 0 and AlignedDataset.change:    # First, determine if there is an offset. If it is greater than 0, move to the right
                # --------------Calculate the optimal sliding distance--------------------
                AlignedDataset.best_ls,AlignedDataset.best_ls_x = self.huadong_ls(line)
                # ------------------------------------------------------------------------
                AlignedDataset.line_l1 = AlignedDataset.line_l1 + AlignedDataset.best_ls
                AlignedDataset.line_l2 = 108 + AlignedDataset.line_l1
                AlignedDataset.d_ls = 1
            elif line < 0 and AlignedDataset.change:
                # --------------Calculate the optimal sliding distance-------------------
                AlignedDataset.best_ls,AlignedDataset.best_ls_x= self.huadong_ls(line)
                # -----------------------------------------------------------------------
                AlignedDataset.line_l1 = AlignedDataset.line_l1 - AlignedDataset.best_ls
                AlignedDataset.line_l2 = 108 + AlignedDataset.line_l1
                AlignedDataset.d_ls = -1
            else:
                if epoch != 1:   # If it is not the first input, the cumulative result of the last input will be executed
                    AlignedDataset.line_l1 = AlignedDataset.line_l1
                    AlignedDataset.line_l2 = 108 + AlignedDataset.line_l1
                    AlignedDataset.d_ls = 0
                    AlignedDataset.best_ls = 0
                    AlignedDataset.best_ls_x =0
                elif AlignedDataset.change == 0:
                    AlignedDataset.line_l1 = AlignedDataset.line_l1
                    AlignedDataset.line_l2 = 108 + AlignedDataset.line_l1
                    AlignedDataset.d_ls = 0
                    AlignedDataset.best_ls = 0
                    AlignedDataset.best_ls_x = 0
                else:       # If it is the first input, initialization will be performed
                    AlignedDataset.line_l1 = int(self.opt.loadSize_w / 3)
                    AlignedDataset.line_l2 = 108 + AlignedDataset.line_l1
                    AlignedDataset.d_ls = 0
                    AlignedDataset.best_ls = 0
                    AlignedDataset.best_ls_x = 0
            # Gate value limit
            if AlignedDataset.line_l1 < 0:
                AlignedDataset.line_l1 = 0
                AlignedDataset.change = 0
            elif AlignedDataset.line_l1 > 212:
                AlignedDataset.line_l1 = 212
                AlignedDataset.change =0
            else:
                AlignedDataset.line_l1 = AlignedDataset.line_l1

            if AlignedDataset.line_l2 < 108:
                AlignedDataset.line_l2 = 108
                AlignedDataset.change = 0
            elif AlignedDataset.line_l2 > 320:
                AlignedDataset.line_l2 = 320
                AlignedDataset.change = 0
            else:
                AlignedDataset.line_l2 = AlignedDataset.line_l2

            AlignedDataset.l1_results = np.append(AlignedDataset.l1_results, AlignedDataset.line_l1)
            AlignedDataset.l2_results = np.append(AlignedDataset.l2_results, AlignedDataset.line_l2)
            if AlignedDataset.l1_results.size <= 1:
                AlignedDataset.l1_results_3 = AlignedDataset.l1_results.copy()  # If the number of elements is less than or equal to 3, the value is assigned as "results"
                AlignedDataset.l2_results_3 = AlignedDataset.l2_results.copy()  # If the number of elements is less than or equal to 3, the value is assigned as "results"
            else:
                AlignedDataset.l1_results_3 = AlignedDataset.l1_results[-1:]  # If the number of elements is greater than 3, the latest three values of "results" will be assigned
                AlignedDataset.l2_results_3 = AlignedDataset.l2_results[-1:]  # If the number of elements is greater than 3, the latest three values of "results" will be assigned

            self.line_l1 = AlignedDataset.l1_results_3[-1]
            self.line_l2 = AlignedDataset.l2_results_3[-1]


        elif two == 1 and len(AlignedDataset.l1_results_3):
            self.line_l1 = AlignedDataset.l1_results_3[-1]
            self.line_l2 = AlignedDataset.l2_results_3[-1]

        elif two == 2:
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
        print('self.l2_results_3', AlignedDataset.l2_results_3)

        print('l1：', self.line_l1)
        print('l2：', self.line_l2)

        return self.line_l1, self.line_l2, AlignedDataset.best_ls_x, AlignedDataset.d_ls

    def __getitem__(self, index):


        AB_path = self.AB_paths[index]    # Read the picture

        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize_w, self.opt.loadSize_h), Image.BICUBIC)
        B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize_w, self.opt.loadSize_h), Image.BICUBIC)

        # Crop A to obtain a and b
        A_w,A_h = A.size
        # l1 = int((A_w+1)/3)
        # l2 = int(A_w -l1)

        l1 = self.line_l1
        l2 = self.line_l2

        a = A.crop((0, 0, l2, A_h)).resize((256, 256),Image.BICUBIC)
        b = A.crop((l1, 0, A_w, A_h),).resize((256, 256),Image.BICUBIC)
        Ba = B.crop((0, 0, l2, A_h)).resize((256, 256),Image.BICUBIC)
        Bb = B.crop((l1, 0, A_w, A_h)).resize((256, 256),Image.BICUBIC)
        Aa = A.crop((0, 0, l2, A_h)).resize((256, 256),Image.BICUBIC)
        Ab = A.crop((l1, 0, A_w, A_h)).resize((256, 256),Image.BICUBIC)

        a = np.array(a)
        b = np.array(b)
        Ba = np.array(Ba)
        Bb = np.array(Bb)
        A = np.array(A)
        B = np.array(B)
        Aa = np.array(Aa)
        Ab = np.array(Ab)

        a = Image.fromarray(np.uint8(a)).convert('RGB')
        b = Image.fromarray(np.uint8(b)).convert('RGB')
        Ba = Image.fromarray(np.uint8(Ba)).convert('RGB')
        Bb = Image.fromarray(np.uint8(Bb)).convert('RGB')
        A = Image.fromarray(np.uint8(A)).convert('RGB')
        B = Image.fromarray(np.uint8(B)).convert('RGB')
        Aa = Image.fromarray(np.uint8(Aa)).convert('RGB')
        Ab = Image.fromarray(np.uint8(Ab)).convert('RGB')



        a = transforms.ToTensor()(a)  # (0 - 255) --> (0 - 1)
        b = transforms.ToTensor()(b)
        Ba = transforms.ToTensor()(Ba)  # (0 - 255) --> (0 - 1)
        Bb = transforms.ToTensor()(Bb)
        A = transforms.ToTensor()(A)  # (0 - 255) --> (0 - 1)
        B = transforms.ToTensor()(B)
        Aa = transforms.ToTensor()(Aa)  # (0 - 255) --> (0 - 1)
        Ab = transforms.ToTensor()(Ab)

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
        # Initialize the minimum absolute value and the corresponding ls, x
        Xl_l1 = self.line_l1
        Xl_l2 = self.line_l2
        min_e_abs = float('inf')
        best_ls = None
        best_x = None

        if line != 0:
            for ls in range(ls_min, ls_max):
                for x in range(x_min, x_max):
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

                    if abs(e) < min_e_abs and e != 0:
                        min_e_abs = abs(e)
                        best_ls = ls
                        best_x = x
        else:
            best_ls = 0
            best_x = 0
        # Print the result
        best_ls_x = best_ls + best_x
        print(f"The smallest absolute value of e: {min_e_abs},ls: {best_ls}, x: {best_x},ls_x:{best_ls_x}")
        return best_ls,best_ls_x
