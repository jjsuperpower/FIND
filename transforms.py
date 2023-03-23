import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from utils import View

from PIL import Image
import os

class Luminance(object):
    def __init__(self, factor:float = 1):
        self.factor = factor

    def __call__(self, img:torch.tensor):
        return torch.clip(img * self.factor, 0, 1)
        
    def __repr__(self):
        return self.__class__.__name__ + f'(factor={self.factor})'
    
class Brightness_Contrast(object):
    def __init__(self, brighness:float = 0, contrast:float = 1):
        self.brightness = brighness
        self.contrast = contrast
    
    def __call__(self, img:torch.tensor):
        img = self.contrast*(img - 0.5) + 0.5 + self.brightness
        return torch.clip(img, 0, 1)
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(brightness={self.brightness}, contrast={self.contrast})'
    
class Hist_EQ():
    def __init__(self):
        pass
    
    def __call__(self, img:torch.tensor):
        img = img.numpy()
        img = img.swapaxes(0, 2).swapaxes(0, 1)       # make channels last
        img = np.clip((img * 255), 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img[:,:,2] = cv2.equalizeHist(img[:,:,2])
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = img.astype(np.float32) / 255
        img = img.swapaxes(0, 1).swapaxes(0, 2)       # make channels first
        img = torch.from_numpy(img)
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    
    
if __name__ == '__main__':
    # check if test image exist
    if not os.path.exists('test.jpg'):
        print('test image not found, downloading...')
        # get picture of cat
        os.system('wget https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg -O test.jpg')
        
    
    # load image
    img = np.asarray(Image.open('test.jpg'))
    
    dark_img = transforms.Compose([transforms.ToTensor(), Luminance(1/8)])(img)
    View.compare_color(img, dark_img)
    
    hist_img = transforms.Compose([Hist_EQ()])(dark_img)
    View.compare_color(dark_img, hist_img)
    
    low_contrast_img = transforms.Compose([transforms.ToTensor(), Brightness_Contrast(-.25, 0.1)])(img)
    View.compare_color(img, low_contrast_img)
    
    