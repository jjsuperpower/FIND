import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchmetrics
import os
from copy import deepcopy

from .transforms import Luminance, Hist_EQ, Brightness_Contrast, Retinex, Blur, HistRemap, DistRemap, Fog, Rain
from .coco_ds import CocoDataset
from .myutils import ImgUtils

class Model_Wrapper(pl.LightningModule):
    def __init__(self, model, model_type='class', scale=1.0):
        super().__init__()
        self.model = model
        
        self.model_type = model_type
        self.scale = scale
        
    def forward(self, x):
        return self.model(x*self.scale)
    
    def eval(self):
        pass

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = F.cross_entropy(y_hat, y)
    #     return loss

    def test_step(self, batch, batch_idx):
        if self.model_type == 'class':
            imgs, y = batch
            y_hat = self(imgs)
            
            num_classes = y_hat.size(1)
            acc1 = torchmetrics.functional.accuracy(y_hat, y, top_k=1, task='multiclass', num_classes=num_classes)
            acc5 = torchmetrics.functional.accuracy(y_hat, y, top_k=5, task='multiclass', num_classes=num_classes)
            
            conf = torch.mean(torch.max(F.softmax(y_hat, dim=1), dim=1).values)
            loss = F.cross_entropy(y_hat, y)
            
            self.log('Top 1 Acc %', acc1*100)
            self.log('Top 5 Acc %', acc5*100)
            self.log('Confidence %', conf*100)
            # self.log('Loss', loss)
            
        elif self.model_type == 'yolo':          
            ids, imgs, orig_sizes = batch
            
            self.model.reset_coco_evaluator()
            evaluator, _ = self.model.predict_coco(ids, imgs, orig_sizes, gen_plot=False)
            results = evaluator.eval()
            
            self.log('mAP 50:95 %', results['AP'] * 100)
            self.log('Confidence %', results['conf'] * 100)
            
        elif self.model_type == 'obj':
            raise NotImplementedError
        
        else:
            raise NotImplementedError

        # mus = torch.zeros(imgs.size()[0:2])
        # sigmas = torch.zeros(imgs.size()[0:2])
        
        # for i in range(imgs.size(0)):
        #     for j in range(imgs.size(1)):
        #         mus[i,j] = torch.mean(imgs[i,j])
        #         sigmas[i,j] = torch.std(imgs[i,j])
        
        mu, sigma = ImgUtils.get_mean_std(imgs)
    
        self.log('Pixel Val MEAN', mu * 255)
        self.log('Pixel Val STD', sigma * 255)
        # self.log('Loss', loss)get_loader()
        
    def plot(self, *args, **kwargs):
        return self.model.plot(*args, **kwargs)
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
class Preprocess():
    def __init__(self, 
                 dataset_path:str, 
                 input_size:tuple, 
                 dataset_type='imagenet', 
                 split='val', 
                 batch_size=128, 
                 num_workers=8, 
                 shuffle=False):
        
        self.dataset_path = dataset_path
        self.input_size = input_size
        
        self.dataset_type = dataset_type
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        self.basic_trans = [
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
        ]
        
        self.current_trans = deepcopy(self.basic_trans)
        
    def basic(self):
        return transforms.Compose(deepcopy(self.basic_trans))
    
    def basic_loader(self):
        tmp_trans = deepcopy(self.current_trans)
        self.current_trans = deepcopy(self.basic_trans)
        loader = self.get_loader()
        self.current_trans = tmp_trans
        return loader
    
    def copy(self):
        return deepcopy(self)
    
    def get_trans(self):
        return transforms.Compose(deepcopy(self.current_trans))
    
    def reset_trans(self):
        self.current_trans = deepcopy(self.basic_trans)
        return self
        
    def luminance(self, factor):
        self.current_trans += [Luminance(factor)]
        return self
    
    def brightness_contrast(self, *args, **kwargs):
        self.current_trans += [Brightness_Contrast(*args, **kwargs)]
        return self
    
    def blur(self, kernel_size):
        self.current_trans += [Blur(kernel_size)]
        return self
    
    def sharpen(self, factor):
        self.current_trans += [transforms.RandomAdjustSharpness(factor, p=1)]
        return self
    
    def hist_eq(self):
        self.current_trans += [Hist_EQ()]
        return self
    
    def retinex(self, *args, **kwargs):
        self.current_trans += [Retinex(*args, **kwargs)]
        return self
    
    def hist_remap(self, *args, **kwargs):
        self.current_trans += [HistRemap(*args, **kwargs)]
        return self
    
    def dist_remap(self, *args, **kwargs):
        self.current_trans += [DistRemap(*args, **kwargs)]
        return self
    
    def fog(self, *args, **kwargs):
        self.current_trans += [Fog(*args, **kwargs)]
        return self
        
    def rain(self, *args, **kwargs):
        self.current_trans += [Rain(*args, **kwargs)]
        return self
        
        
    def get_loader(self):
        return DataLoader(self.get_dataset(), 
                          batch_size=self.batch_size, 
                          shuffle=self.shuffle, 
                          num_workers=self.num_workers)
        
    def get_dataset(self):
        if self.dataset_type == 'imagenet':
            return datasets.ImageNet(root=self.dataset_path, split=self.split, transform=self.get_trans())
        elif self.dataset_type == 'coco':
            return CocoDataset(root=self.dataset_path, set='val2017', transform=self.get_trans())
        else:
            raise NotImplementedError