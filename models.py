import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchmetrics

from copy import deepcopy

from transforms import Luminance, Hist_EQ, Brightness_Contrast, Retinex, Blur

class Model_Wrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        acc1 = torchmetrics.functional.accuracy(y_hat, y, top_k=1)
        acc5 = torchmetrics.functional.accuracy(y_hat, y, top_k=5)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_acc1', acc1)
        self.log('test_acc5', acc5)
        self.log('test_loss', loss)
        
        return loss

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
    
    def get_trans(self):
        return transforms.Compose(deepcopy(self.current_trans))
    
    def reset_trans(self):
        self.current_trans = deepcopy(self.basic_trans)
        return self
        
    def luminance(self, factor):
        self.current_trans += [Luminance(factor)]
        return self
    
    def brightness_contrast(self, brightness, contrast):
        self.current_trans += [Brightness_Contrast(brightness, contrast)]
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
        
    def get_loader(self):
        return DataLoader(self.get_dataset(), 
                          batch_size=self.batch_size, 
                          shuffle=self.shuffle, 
                          num_workers=self.num_workers)
        
    def get_dataset(self):
        if self.dataset_type == 'imagenet':
            return datasets.ImageNet(root=self.dataset_path, split=self.split, transform=self.get_trans())