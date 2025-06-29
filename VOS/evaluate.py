r'''
计算psnr,ssim,lpips
'''
import os
import cv2
import lpips
import argparse
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import sys
from skimage import io, color, filters

def LPIPS_0(real_B_dis,fake_B_dis):
    loss_fn = lpips.LPIPS(net='alex', version='0.1')
    # the total list of images
    real_B= lpips.im2tensor(lpips.load_image(real_B_dis))
    fake_B = lpips.im2tensor(lpips.load_image(fake_B_dis))
    current_lpips_distance = loss_fn.forward(real_B, fake_B)
    return current_lpips_distance


def rmetrics(a,b):
    
    #pnsr
    mse = np.mean((a-b)**2)
    #psnr = 10*math.log10(1/mse)
    psnr = 10 * np.log10(255 * 255 / mse)

    #ssim
    ssim = compare_ssim(a,b,win_size=3,channel_axis=2,data_range=255)
    #ssim = compare_ssim(a, b, win_size=11, channel_axis=2, data_range=255)

    return psnr, ssim


def P_S_L(result_path):
    global fake_a,real_Ba,fake_b,real_Bb

    result_path = result_path
    reference_path = result_path

    result_dirs = os.listdir(result_path)


    sumpsnr_a, sumssim_a ,sumlpips_a= 0.,0.,0.
    sumpsnr_b, sumssim_b, sumlpips_b = 0., 0., 0.

    lpips_dis_fake_a ,lpips_dis_real_Ba,lpips_dis_real_b,lpips_dis_real_Bb = 0,0,0,0

    N , i = 0 , 0
    for imgdir in result_dirs:
        if '.png' in imgdir:
            #reference image
            i = i + 1
            if '_fake_a.png' in imgdir:
                fake_a = io.imread(os.path.join(reference_path,imgdir))
                lpips_dis_fake_a = os.path.join(reference_path,imgdir)
                fake_a = cv2.cvtColor(fake_a, cv2.COLOR_BGR2RGB)
            elif '_real_Ba.png' in imgdir:
                real_Ba = io.imread(os.path.join(result_path, imgdir))
                lpips_dis_real_Ba = os.path.join(reference_path, imgdir)
                real_Ba = cv2.cvtColor(real_Ba, cv2.COLOR_BGR2RGB)
            elif '_fake_b.png' in imgdir:
                fake_b = io.imread(os.path.join(result_path, imgdir))
                lpips_dis_real_b = os.path.join(reference_path, imgdir)
                fake_b = cv2.cvtColor(fake_b, cv2.COLOR_BGR2RGB)
            elif '_real_Bb.png' in imgdir:
                real_Bb = io.imread(os.path.join(result_path, imgdir))
                lpips_dis_real_Bb = os.path.join(reference_path, imgdir)
                real_Bb = cv2.cvtColor(real_Bb, cv2.COLOR_BGR2RGB)
            elif '_real_Ba.png' in imgdir:
                print('The calculation of the {} sheet is completed'.format((i-3)/6+1))

            if i%9 == 0:
                psnr_a, ssim_a = rmetrics(fake_a,real_Ba)
                psnr_b, ssim_b = rmetrics(fake_b, real_Bb)
                lpips_a = LPIPS_0(lpips_dis_fake_a,lpips_dis_real_Ba)
                lpips_b = LPIPS_0(lpips_dis_real_Bb, lpips_dis_real_b)

                sumpsnr_a += psnr_a
                sumssim_a += ssim_a
                sumlpips_a +=lpips_a

                sumpsnr_b += psnr_b
                sumssim_b += ssim_b
                sumlpips_b += lpips_b

                N +=1

                with open(os.path.join(result_path,'metrics_{}.txt'.format('two.side')), 'a') as f:
                    f.write('{}: psnr_a={} ssim_a={} lpips_a{} psnr_b={} ssim_b={} lpips_b{}\n'.format(N,psnr_a,ssim_a,lpips_a,psnr_b,ssim_b,lpips_b))
                    print('{}: avpsnr_a={} avssim_a={} avlpips_a{} avpsnr_b={} avssim_b={} avlpips_b{}\n'.format(N,sumpsnr_a/N,sumssim_a/N,sumlpips_a/N,sumpsnr_b/N,sumssim_b/N,sumlpips_b/N))

    mpsnr_a = sumpsnr_a/N
    mssim_a = sumssim_a/N
    mlpips_a =sumlpips_a/N
    mpsnr_b = sumpsnr_b / N
    mssim_b = sumssim_b / N
    mlpips_b = sumlpips_b / N

    with open(os.path.join(result_path,'metrics.txt'), 'a') as f:
        f.write('Average: avpsnr_a={} avssim_a={} avlpips_a{} avpsnr_b={} avssim_b={} avlpips_b{}\n'.format(mpsnr_a, mssim_a,mlpips_a,mpsnr_b, mssim_b,mlpips_b))
        print('-------------------------------------------------------------------------')
        print('Average: avpsnr_a={} avssim_a={} avlpips_a{} avpsnr_b={} avssim_b={} avlpips_b{}\n'.format(mpsnr_a, mssim_a,mlpips_a,mpsnr_b, mssim_b,mlpips_b))
    return mpsnr_a,mssim_a,mlpips_a,mpsnr_b,mssim_b,mlpips_b


def PSNR_SSIM_LPIPS(result_path):
    PSNR_a ,SSIM_a,LPIPS_a,PSNR_b,SSIM_b,LPIPS_b= P_S_L(result_path)
    print('-------------------------------------------------------------------------')
    # print('Average: avpsnr_a={} avssim_a={} avlpips_a{} avpsnr_b={} avssim_b={} avlpips_b{}\n'.format(PSNR_a, SSIM_a,LPIPS_a,PSNR_b, SSIM_b,LPIPS_b))
    return  PSNR_a,PSNR_b, SSIM_a,SSIM_b , LPIPS_a, LPIPS_b
