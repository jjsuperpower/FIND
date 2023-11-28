import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os

from .myutils import View, ImgUtils

class Luminance(object):
    def __init__(self, factor:float = 1):
        self.factor = factor

    def __call__(self, img:torch.Tensor):
        return torch.clip(img * self.factor, 0, 1)
        
    def __repr__(self):
        return self.__class__.__name__ + f'(factor={self.factor})'
    
class Brightness_Contrast(object):
    def __init__(self, brighness:float = 0, contrast:float = 1, no_clip:bool = False):
        self.brightness = brighness
        self.contrast = contrast
        self.no_clip = no_clip
    
    def __call__(self, img:torch.Tensor):
        img = self.contrast*(img - 0.5) + 0.5 + self.brightness
        
        if self.no_clip:
            if img.min() < 0:
                img = img - img.min()
            img = img / img.max()
            
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
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img[:,:,2] = cv2.equalizeHist(img[:,:,2])
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img[:,:,0] = cv2.equalizeHist(img[:,:,0])
        # img[:,:,1] = cv2.equalizeHist(img[:,:,1])
        # img[:,:,2] = cv2.equalizeHist(img[:,:,2])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        img = img.swapaxes(0, 1).swapaxes(0, 2)       # make channels first
        img = torch.from_numpy(img)
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class AHE():
    def __init__(self, tile_size:int, clip_limit:float = 2.0):
        self.tile_size = tile_size
        self.clip_limit = clip_limit
    
    def __call__(self, img:torch.Tensor):
        img = img.numpy()
        img = img.swapaxes(0, 2).swapaxes(0, 1)       # make channels last
        img = np.clip((img * 255), 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.tile_size, self.tile_size))
        img[:,:,2] = clahe.apply(img[:,:,2])
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
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
    
    
class HistRemap():
    def __init__(self, target_hist:torch.Tensor):
        self.target_hist = target_hist
        
    def __call__(self, img:torch.Tensor):
        orig_img = ImgUtils.rgb_to_hsv(img.clone())
        orig_img_type = orig_img.dtype
        
        img = (img[2,...] * 255).type(torch.uint8)
    
        lut = torch.zeros(shape=(256), dtype=torch.uint8)
    
        # first get the histogram of the image
        img_hist = ImgUtils.img_to_hist(img)
        hist = self.target_hist
        img_hist = img_hist.astype(np.uint64)
        
        hist_idx = np.nonzero(hist)[0][0]
        lut_idx = 0
        diff = 0
        
        while(lut_idx < 256 and hist_idx < 256):       
            diff = hist[hist_idx] - img_hist[lut_idx] + diff
            lut[lut_idx] = hist_idx
            
            if diff < 0:
                diff = diff + img_hist[lut_idx]
                hist_idx += 1
            
            elif diff > 0:
                diff = diff - hist[hist_idx]
                lut_idx += 1
            
            else:
                hist_idx += 1
                lut_idx += 1   

        if hist_idx >= 255:
            lut[lut_idx:] = 255
        else:
            lut[lut_idx:] = hist_idx
            
        # apply lut to new image
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                img[x, y] = lut[img[x, y]]
                
        new_img = orig_img
        new_img[2,...] = img / 255
                
        return ImgUtils.hsv_to_rgb(new_img).type(orig_img_type)
    
    def __repr__(self):
        return self.__class__.__name__ + '()'

class DistRemap:
    ''' Transform for remapping images to a target distribution '''
    
    def __init__(self, mu:torch.Tensor=None, sigma:torch.Tensor=None, no_clip:bool=False):
        self.mu = mu
        self.sigma = sigma
        self.no_clip = no_clip
        
    def __call__(self, img:torch.TensorType) -> torch.Tensor:
        ''' Mu and sigma are in range [0, 1]'''
        img_mu, img_sigma = ImgUtils.get_mean_std(img)
        
        if self.mu is None:
            self.mu = img_mu
        if self.sigma is None:
            self.sigma = img_sigma
        
        brightness = (self.mu - img_mu)
        contrast = self.sigma / img_sigma
        
        img = (img - 0.5) * contrast + 0.5 + brightness
        
        if self.no_clip:
            if img.min() < 0:
                img = img - img.min()
            img = img / img.max()
            
        return torch.clip(img, 0, 1)
    
class Fog():
    def __init__(self, level:int):
        self.level = level
        self.bc = Brightness_Contrast(abs(level), 1, no_clip=True)
        
    def __call__(self, img:torch.Tensor):
        if self.level < 0:
            img = -img + 1
            img = self.bc(img)
            img = -img + 1
            
        else:
            img = self.bc(img)
            
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + f'(level={self.level})'
    
    
class Rain():
    def __init__(self, level:int, threshold:float=80):
        self.level = level
        self.threshold = threshold
        self.fog = Fog(-level)
        
    def __call__(self, img:torch.Tensor):
        
        # convert to hsv
        img = ImgUtils.rgb_to_hsv(img)
        v_chan = img[2,...]
        img = ImgUtils.hsv_to_rgb(img)
        keep_pixels = torch.nonzero(v_chan > (self.threshold / 100))
        
        new_img = self.fog(img.clone())
        new_img[:, keep_pixels[:, 0], keep_pixels[:, 1]] = img[:, keep_pixels[:, 0], keep_pixels[:, 1]]
        
        return torch.clip(new_img, 0, 1)        # remove negative zero
    
    def __repr__(self):
        return self.__class__.__name__ + f'(level={self.level})'
        
    
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
    
    dark_img = Luminance(1/8)(img)
    View.compare_color([img, dark_img],['Original', 'Dark'], figsize=(10, 5))
    
    hist_img = Hist_EQ()(dark_img)
    View.compare_color([dark_img, hist_img],['Dark', 'Hist EQ'], figsize=(10, 5))
    
    ahe_img = AHE(8, clip_limit=2)(img)
    View.compare_color([img, ahe_img],['Original', 'AHE'], figsize=(10, 5))
    
    low_contrast_img = Brightness_Contrast(-.25, 0.2)(img)
    View.compare_color([img, low_contrast_img],['Original', 'Low Contrast'], figsize=(10, 5))
    
    retinex_img = Retinex('SSR', 100)(low_contrast_img)
    View.compare_color([low_contrast_img, retinex_img],['Low Contrast', 'Retinex'], figsize=(10, 5))
    
    blur_img = Blur(11)(img)
    View.compare_color([img, blur_img],['Original', 'Blur'], figsize=(10, 5))
    
    