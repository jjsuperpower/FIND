import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from utils import View

from PIL import Image
import os
import sys

class Luminance(object):
    def __init__(self, factor:float = 1):
        self.factor = factor

    def __call__(self, img:torch.Tensor):
        return torch.clip(img * self.factor, 0, 1)
        
    def __repr__(self):
        return self.__class__.__name__ + f'(factor={self.factor})'
    
class Brightness_Contrast(object):
    def __init__(self, brighness:float = 0, contrast:float = 1):
        self.brightness = brighness
        self.contrast = contrast
    
    def __call__(self, img:torch.Tensor):
        img = self.contrast*(img - 0.5) + 0.5 + self.brightness
        return torch.clip(img, 0, 1)
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(brightness={self.brightness}, contrast={self.contrast})'
    
class Hist_EQ():
    def __init__(self):
        pass
    
    def __call__(self, img:torch.Tensor):
        img = img.numpy()
        img = img.swapaxes(0, 2).swapaxes(0, 1)       # make channels last
        img = np.clip((img * 255), 0, 255).astype(np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # img[:,:,2] = cv2.equalizeHist(img[:,:,2])
        # img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img[:,:,0] = cv2.equalizeHist(img[:,:,0])
        img[:,:,1] = cv2.equalizeHist(img[:,:,1])
        img[:,:,2] = cv2.equalizeHist(img[:,:,2])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        img = img.swapaxes(0, 1).swapaxes(0, 2)       # make channels first
        img = torch.from_numpy(img)
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
class Retinex():
    def __init__(self, algorithm:str, *args, **kwargs):
        # check of retinex folder exists
        if not os.path.exists('retinex'):
            print('retinex submodule not found')
            print('Please run `git submodule update --init --recursive` to download retinex')
            raise FileNotFoundError('retinex folder not found')
    
        # add retinex to path
        # workdir_path = os.path.dirname(os.path.abspath(__file__))
        # sys.path.append(os.path.join(workdir_path, 'retinex'))
        
        self.args = args
        self.kwargs = kwargs
        
        # import retinex_MSRCP from code folder in retinex submodule        
        from retinex.code.retinex import retinex_SSR, retinex_MSR, retinex_MSRCR, retinex_gimp, retinex_MSRCP, retinex_AMSR
        
        if algorithm == 'SSR':
            self.retinex_fn = retinex_SSR
        elif algorithm == 'MSR':
            self.retinex_fn = retinex_MSR
        elif algorithm == 'MSRCR':
            self.retinex_fn = retinex_MSRCR
        elif algorithm == 'gimp':
            self.retinex_fn = retinex_gimp
        elif algorithm == 'MSRCP':
            self.retinex_fn = retinex_MSRCP
        elif algorithm == 'AMSR':
            self.retinex_fn = retinex_AMSR
        else:
            raise ValueError(f'algorithm {algorithm} not found')
            
    def __call__(self, img:torch.Tensor):
        img = img.numpy() * 255
        img = img.swapaxes(0, 2).swapaxes(0, 1)       # make channels last
        img = img.astype(np.uint8)
        img = self.retinex_fn(img, *self.args, **self.kwargs)
        img = img.astype(np.float32) / 255
        img = img.swapaxes(0, 1).swapaxes(0, 2)       # make channels first
        img = torch.from_numpy(img)
        return torch.clip(img, 0, 1)
        
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class Blur():
    def __init__(self, kernel_size:int = 3):
        
        # raise error if kernel size is not odd
        if kernel_size % 2 == 0:
            raise ValueError('kernel size should be odd')
        
        self.kernel_size = kernel_size
        conv2d = nn.Conv2d(1, 1, kernel_size, padding='same', bias=False)
        conv2d.weight.requires_grad = False
        conv2d.weight[...] = 1 / (kernel_size**2)
        self.conv2d = conv2d
        
    def __call__(self, img:torch.Tensor):
        img_type = img.dtype
        img = img.float()
        img = img.unsqueeze(0)
        img = img.swapaxes(0, 1)
        img = self.conv2d(img)
        img = img.swapaxes(0, 1)
        img = img.squeeze(0)
        return torch.clip(img.type(img_type), 0, 1)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(kernel_size={self.kernel_size})'
    
if __name__ == '__main__':
    # check if test image exist
    if not os.path.exists('test.jpg'):
        print('test image not found, downloading...')
        # get picture of cat
        os.system('wget https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg -O test.jpg')
        
    
    # load image
    img = Image.open('test.jpg')
    img = transforms.Resize((256, 256))(img)
    img = transforms.ToTensor()(img)
    
    # dark_img = Luminance(1/8)(img)
    # View.compare_color(img, dark_img)
    
    # hist_img = Hist_EQ()(dark_img)
    # View.compare_color(dark_img, hist_img)
    
    # low_contrast_img = Brightness_Contrast(-.25, 0.2)(img)
    # View.compare_color(img, low_contrast_img)
    
    # retinex_img = Retinex()(low_contrast_img)
    # View.compare_color(low_contrast_img, retinex_img)
    
    blur_img = Blur(11)(img)
    View.compare_color(img, blur_img)
    
    