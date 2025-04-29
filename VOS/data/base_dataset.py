import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

'''
定义一些公用的属性/函数；一般的，torch.utils.data.Dataset 本身已经包含了很多属性，如 __len__, __getitem__ 等。
一般我们会新增一个成员函数 name 和 initialize，分别用于：
1）name：没有任何意义，纯属装 B
2）在 pytorch 中，我们经常会使用到 parser，即一个能够从命令行赋予超参数值的辅助类，我们在代码中实例化它的一个对象为 "opt" ，而且，诸如 opt.img_size, opt.batch_size 这样的参数是与 data 相关的，所以我们通常会在这个函数引入 opt，并将它作为自己一个属性 self.opt，如此，我们就可以随时访问所有的超参数了。
'''
class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__() # 使用当前类的实例 self 作为第一个参数,调用 BaseDataset 类的构造函数。 super表示继承父类，表示继承Dataset中的初始化函数进行初始化

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
