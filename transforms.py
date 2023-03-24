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
    def __init__(self, sigmas=[12,80,250],s1=0.01,s2=0.01):
        self.sigmas = sigmas
        self.s1 = s1
        self.s2 = s2
        
    # ---------------------------------------------------------------------
    # begin of code taken from
    # https://github.com/muggledy/retinex/
    
        self.eps=np.finfo(np.double).eps
    
    def get_gauss_kernel(self, sigma,dim=2):
        '''1D gaussian function: G(x)=1/(sqrt{2π}σ)exp{-(x-μ)²/2σ²}. Herein, μ:=0, after 
        normalizing the 1D kernel, we can get 2D kernel version by 
        matmul(1D_kernel',1D_kernel), having same sigma in both directions. Note that 
        if you want to blur one image with a 2-D gaussian filter, you should separate 
        it into two steps(i.e. separate the 2-D filter into two 1-D filter, one column 
        filter, one row filter): 1) blur image with first column filter, 2) blur the 
        result image of 1) with the second row filter. Analyse the time complexity: if 
        m&n is the shape of image, p&q is the size of 2-D filter, bluring image with 
        2-D filter takes O(mnpq), but two-step method takes O(pmn+qmn)'''
        ksize=int(np.floor(sigma*6)/2)*2+1 #kernel size("3-σ"法则) refer to 
        #https://github.com/upcAutoLang/MSRCR-Restoration/blob/master/src/MSRCR.cpp
        k_1D=np.arange(ksize)-ksize//2
        k_1D=np.exp(-k_1D**2/(2*sigma**2))
        k_1D=k_1D/np.sum(k_1D)
        if dim==1:
            return k_1D
        elif dim==2:
            return k_1D[:,None].dot(k_1D.reshape(1,-1))
        
    def gauss_blur_original(self, img,sigma):
        '''suitable for 1 or 3 channel image'''
        row_filter=self.get_gauss_kernel(sigma,1)
        t=cv2.filter2D(img,-1,row_filter[...,None])
        return cv2.filter2D(t,-1,row_filter.reshape(1,-1))

        
    def simplest_color_balance(self, img_msrcr, s1, s2):
        '''see section 3.1 in “Simplest Color Balance”(doi: 10.5201/ipol.2011.llmps-scb). 
        Only suitable for 1-channel image'''
        sort_img=np.sort(img_msrcr,None)
        N=img_msrcr.size
        Vmin=sort_img[int(N*s1)]
        Vmax=sort_img[int(N*(1-s2))-1]
        img_msrcr[img_msrcr<Vmin]=Vmin
        img_msrcr[img_msrcr>Vmax]=Vmax
        return (img_msrcr-Vmin)*255/(Vmax-Vmin)
    
    def retinex_MSRCP(self, img, sigmas, s1, s2):
        '''compare to others, simple and very fast'''
        Int=np.sum(img,axis=2)/3
        Diffs=[]
        for sigma in sigmas:
            Diffs.append(np.log(Int+1)-np.log(self.gauss_blur_original(Int,sigma)+1))
        MSR=sum(Diffs)/3
        Int1=self.simplest_color_balance(MSR,s1,s2)
        B=np.max(img,axis=2)
        A=np.min(np.stack((255/(B+self.eps),Int1/(Int+self.eps)),axis=2),axis=-1)
        return (A[...,None]*img).astype(np.float32)
    
    # ---------------------------------------------------------------------
    
    def __call__(self, img:torch.Tensor):
        img = img.numpy() * 255
        img = self.retinex_MSRCP(img, self.sigmas, self.s1, self.s2)
        img = torch.from_numpy(img)
        return torch.clip(img, 0, 1)
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    
if __name__ == '__main__':
    # check if test image exist
    if not os.path.exists('test.jpg'):
        print('test image not found, downloading...')
        # get picture of cat
        os.system('wget https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg -O test.jpg')
        
    
    # load image
    img = np.asarray(Image.open('test.jpg')) / 255
    img = transforms.ToTensor()(img)
    
    dark_img = Luminance(1/8)(img)
    View.compare_color(img, dark_img)
    
    hist_img = Hist_EQ()(dark_img)
    View.compare_color(dark_img, hist_img)
    
    low_contrast_img = Brightness_Contrast(-.25, 0.1)(img)
    View.compare_color(img, low_contrast_img)
    
    retinex_img = Retinex()(low_contrast_img)
    View.compare_color(low_contrast_img, retinex_img)
    
    