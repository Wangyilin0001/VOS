from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array

def tensor2im(input_image, imtype=np.uint8):     # 输入的image_numpy应该数值范围为-1-1之间，使用之前用Tahn激活一下即可
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    # if image_numpy.shape[0] == 1:                         #如果输入图像只有单通道,则复制三次以匹配 RGB 格式
    #     image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0      # 将图像数据从 PyTorch 的 CHW 格式转换为 OpenCV 的 HWC 格式,并对数据进行归一化
    image_numpy = np.clip(image_numpy, 0.0, 255.0)                              # 最后,使用 np.clip() 函数确保图像数据在 0 ~ 255 的范围内,并将数据类型转换为 imtype 指定的类型(默认为 np.uint8):
    return image_numpy.astype(imtype)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
