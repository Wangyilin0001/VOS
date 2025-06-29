# -*- coding: utf-8 -*-
import time

import cv2
import numpy as np

from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from models.pix2pix_model import Pix2PixModel
from models.VOS_model import VOSModel
from options.test_options import TestOptions
import math
import csv
import ntpath
import os
from util import util
from tqdm import tqdm
from util.visualizer import save_images
from util import html
import evaluate

#  python train.py --which_epoch 0
#  python -m visdom.server -p 8097

if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt2 = TestOptions().parse()

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    total_iters = 0  # the total number of training iterations
    line = 0
    Epoch_s = 0


    # Output address check
    if os.path.exists(opt.train_path) is not True:
        os.makedirs(opt.train_path)
    if os.path.exists(opt.train_path) is not True:
        os.makedirs(opt.test_path)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        t_data = 0

        # Initialize to reset the evaluation metrics to zero
        ssim_sum_a = 0.0
        psnr_sum_a = 0.0
        ssim_sum_b = 0.0
        psnr_sum_b = 0.0

        # Load data
        data_loader,l1,l2,best_ls_x,d_ls = CreateDataLoader(opt,line,epoch)
        dataset = data_loader.load_data()
        data_loader_test,l1,l2,best_ls_x,d_ls = CreateDataLoader(opt2,line,epoch,two=2)  # The second input does not change the size
        dataset_test = data_loader_test.load_data()

        line = 0  # Clear the previous offset to 0

        dataset_size = len(data_loader)
        dataset_size_test = len(data_loader_test)
        print('#training images = %d' % dataset_size)

        for i, data in enumerate(dataset):

            psnr_a = 0.0
            ssim_a = 0.0
            psnr_b = 0.0
            ssim_b = 0.0

            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:   #  opt.print_freq = 100  Update the time information once every 100 times
                if t_data:
                    next_data = (iter_start_time - t_data-iter_data_time) / 100 * dataset_size
                else:
                    next_data = (iter_start_time - iter_data_time) / 100 * dataset_size
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            line = model.optimize_parameters(l1,l2,line,best_ls_x,d_ls)
            # Return loss
            psnr_a,ssim_a,psnr_b,ssim_b = model.return_PSNR_SSIM()
            # Obtain the total loss
            psnr_sum_a = psnr_a + psnr_sum_a
            ssim_sum_a = ssim_a + ssim_sum_a
            ssim_sum_b = ssim_b + ssim_sum_b
            psnr_sum_b = psnr_b + psnr_sum_b

            total_iters += opt.batchSize
            losses = model.get_current_losses()  # ！！！

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file   opt.display_freq = 100
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()

                # -----------------Obtain the original name-----------------------------------
                img_path = model.get_image_paths()
                short_path = ntpath.basename(img_path[0])
                name = os.path.splitext(short_path)[0]
                # # -----------------Obtain the original name-----------------------------------
                # visuals = model.get_current_visuals()
                # for label, image in visuals.items():
                #     image_numpy = util.tensor2im(image)
                #     util.save_image(image_numpy,
                #                 r'D:\GAN-NET\MUGAN-me\test\test\attion/' + '%s_%s_%d_%d.png' % (
                #                     label , name, total_iters, 3))

                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result,name)


            if total_steps % opt.print_freq == 0:
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data,next_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)


                    # --------------------------Print loss information------------------------------
                    message_loss = ''
                    for k, v in losses.items():
                        message_loss += '%s:%.3f,' % (k, v)
                    f1 = open(r'./each_loss.csv', 'a', newline='')

                    csv_writer = csv.writer(f1)
                    csv_writer.writerow(['epoch:%d,'%epoch+'number:%d,'%i+message_loss])
                    f1.close()
                    #  ----------------------------------------------------------------------------

                    f2 = open(r'./l1l2.csv', 'a', newline='')

                    csv_writer = csv.writer(f2)
                    csv_writer.writerow(['epoch:%d,' % epoch + 'l1:%d,' % l1 + 'l2:%d,' % l2 + 'line:%d,' % line])
                    f2.close()


                    #  --------------------------Print the PSNR and SSIM loss -----------------------------

                    f3 = open(r'./PSNR_SSIM.csv', 'a', newline='')

                    csv_writer = csv.writer(f3)
                    csv_writer.writerow(['psnr_a:%.4f \t ssim_a:%.4f \t psnr_b:%.4f \t ssim_b:%.4f \t Apsnr_sum_a:%.4f \t Assim_sum_a:%.4f \t Apsnr_sum_b:%.4f'
                  ' \t Assim_sum_b:%.4f \t'%(psnr_a,ssim_a,psnr_b,ssim_b,psnr_sum_a/epoch_iter, ssim_sum_a/epoch_iter,
                                          psnr_sum_b/epoch_iter, ssim_sum_b/epoch_iter)])
                    f3.close()

            # 打印每一小轮的训练时间进行观察
            print('epoch:%d \t End of epoch %d / %d \t Time Taken: %.3f sec\t  Time Total: %.3f sec' %
                  (epoch, dataset_size, epoch_iter, (time.time() - iter_start_time) / opt.batchSize,
                   (time.time() - epoch_start_time) / opt.batchSize))

            print('psnr_a:%.4f \t ssim_a:%.4f \t psnr_b:%.4f \t ssim_b:%.4f \t Apsnr_sum_a:%.4f \t Assim_sum_a:%.4f \t Apsnr_sum_b:%.4f'
                  ' \t Assim_sum_b:%.4f \t'%(psnr_a,ssim_a,psnr_b,ssim_b,psnr_sum_a/epoch_iter, ssim_sum_a/epoch_iter,
                                          psnr_sum_b/epoch_iter, ssim_sum_b/epoch_iter))


        if epoch == opt.niter + opt.niter_decay:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))

            model.save_networks('latest')

            iter_data_time = time.time()

        if epoch <= opt.niter + opt.niter_decay and epoch % opt.train_number == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        model.update_learning_rate()
