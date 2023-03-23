import os
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
import lightning.pytorch as pl
import torch.utils.data as data_utils
import numpy as np

from models import Model_Wrapper
from transforms import Darken
from utils import View

PATH_TO_IMAGENET = '../../datasets/imagenet/2012/'
NUM_IMG_EVAL = 10000

# get Resnet50
resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet50 = Model_Wrapper(resnet50)

# create basic imagenet transform
img_trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    Darken(.25),
])
    

# get imagenet dataset
imgnet_val_orig = datasets.ImageNet(root=PATH_TO_IMAGENET, split='val', transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),]))

imgnet_val_set = datasets.ImageNet(root=PATH_TO_IMAGENET, split='val', transform=img_trans)
# imagenet_val = data_utils.Subset(imagenet_val, torch.arange(NUM_IMG_EVAL) & torch.randint(0, 1, (10000)))                       # only use 100,000 images
imgnet_val_loader = DataLoader(imgnet_val_set, batch_size=128, shuffle=False, num_workers=8)

orig, _ = imgnet_val_orig[0]
mod, _ = imgnet_val_set[0]

orig = orig.numpy().reshape((1, *orig.size()))
mod = mod.numpy().reshape((1, *mod.size()))

orig = orig.swapaxes(1, 3).swapaxes(1, 2)
mod = mod.swapaxes(1, 3).swapaxes(1, 2)

View.compare_color(orig, mod)

# use torch lighning to evaluate the model
trainer = pl.Trainer(accelerator="auto")
trainer.test(resnet50, imgnet_val_loader)