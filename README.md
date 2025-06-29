# VOS
VOS：Towards Thermal Infrared Image Colorization via View Overlap Strategy  
# Prerequisites
  * python 3.7  
  * torch 1.13.1  
  * torchvision 0.14.1
  * dominate
  * visdom
# Trian  
`python train.py --dataroot [dataset root] --name [experiment_name] --phase train --which_epoch latest`  
# Test
`python test.py --dataroot [dataset root] --name [experiment_name] --phase test --which_epoch latest`  
# Colorization results  
## FLIR dataset  

## KAITS dataset  

# Acknowledgments
This code heavily borrowes from [pix2pix](https://github.com/phillipi/pix2pix),[TICCGAN](https://github.com/Kuangxd/TICCGAN),[MUGAN](https://github.com/HangyingLiao/MUGAN),[lkat-gan](https://github.com/jinxinhuo/LKAT-GAN-for-Infrared-Image-Colorization) and [FRAGAN](https://github.com/cyanymore/FRAGAN?tab=readme-ov-file#colorization-results).
