import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import io

# set matplotlib to show images with black background
# plt.style.use('dark_background')

class Void(object):
    def write(self, *args, **kwargs):
        pass

class View:
    @staticmethod
    def show_random_color(x, num_to_show):
        plt.figure(figsize=(6,2))
        for i in range(num_to_show):
            ax = plt.subplot(1, num_to_show, i + 1)
            plt.imshow(x[np.random.random_integers(0,x.shape[0])])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def show_all_color(x):
        plt.figure()
        for i in range(len(x)):
            ax = plt.subplot(1, x.shape[0], i + 1)
            plt.imshow(x[i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        
    @staticmethod
    def chan_f2l(array:np.ndarray) -> np.ndarray:
        ''' Move channels from first to last axis
        
        Args: 
            array (np.ndarray): array with shape (C, H, W) or (N, C, H, W)
        
        returns: 
            array with shape (H, W, C) or (N, H, W, C)
        
        raises: 
            ValueError if array does not have 3 or 4 dimensions
        
        '''
        
        if len(array.shape) == 3:
            return np.transpose(array, (1, 2, 0))
        elif len(array.shape) == 4:
            return np.transpose(array, (0, 2, 3, 1))
        else:
            raise ValueError('Array must have 3 or 4 dimensions')
    
    @staticmethod
    def add_dim(array:np.ndarray) -> np.ndarray:
        ''' Add dimension at the beginning of the array
        
        Args: array (np.ndarray)
    
        Returns: array with shape (1, *array.shape)
        
        '''
        return array.reshape((1, *array.shape))
        
        
    @staticmethod
    def compare_color(imgs_list:list[np.ndarray| torch.Tensor]| np.ndarray| torch.Tensor, 
                      labels:list[str]|str=None, 
                      figsize:tuple=(10, 6)) -> None:
        ''' Compare multiple images
        
        This function will plot multiple images in a grid. If there is only one image group,
        it will plot each image group in a row. If there are multiple image groups, it will
        plot multiple rows, each row containing one image group. Each row can have an optional
        title.
        
        All images must have the same shape in the list.
        
        `Args`: 
            imgs_list (list[np.ndarray, torch.Tensor]): list of image groups. Each image group
        
        `raises`: ValueError if imgs_list is empty
        
        '''
        # convert to list if not list
        if not isinstance(imgs_list, list):
            imgs_list = [imgs_list]
            
        # if labels is None, create labels
        if labels is None:
            labels = [f'Image Group {i}' for i in range(len(imgs_list))]
            
        # if imgs_list is empty, raise error
        if len(imgs_list) == 0:
            raise ValueError('Must have at least 1 image')
            
        # if labels is string, convert to list
        if isinstance(labels, str):
            labels = [labels]
            
        # if labels is not the same length as imgs_list, raise error
        if len(labels) != len(imgs_list):
            raise ValueError('Labels must be the same length as imgs_list')
        
        # convert torch tensors to numpy arrays if necessary
        for i in range(len(imgs_list)):
            if isinstance(imgs_list[i], torch.Tensor):
                imgs_list[i] = View.chan_f2l(imgs_list[i].double().numpy())
            
        # make sure all images have 4 dimensions
        for i in range(len(imgs_list)):
            if len(imgs_list[i].shape) == 3:
                imgs_list[i] = View.add_dim(imgs_list[i])
            
        # first axis is rows, second axis is columns
        plt.figure(figsize=figsize)
        fig, axs = plt.subplots(len(imgs_list), len(imgs_list[0]), figsize=figsize)
        
        # if displaying one image
        if (len(imgs_list) == 1) and (len(imgs_list[0]) == 1):
            axs.axis('off')
            axs.imshow(imgs_list[0][0])
            axs.set_title(labels[0], fontsize=10)
            
        # if only one image group, plot each image in a column
        elif len(imgs_list) == 1:
            axs[0].set_title(labels[0], fontsize=10)
            for i, img in enumerate(imgs_list[0]):
                axs[i].axis('off')
                axs[i].imshow(img)

        # if only one image in each image group, plot each image group in a row
        elif len(imgs_list[0]) == 1:
            for i, imgs in enumerate(imgs_list):
                axs[i].axis('off')
                axs[i].set_title(labels[i], fontsize=10)
                axs[i].imshow(imgs[0])
                
        else:
            for i, imgs in enumerate(imgs_list):        # rows
                axs[i, 0].set_title(labels[i], fontsize=10)
                for j, img in enumerate(imgs):          # columns
                    axs[i, j].axis('off')
                    axs[i, j].imshow(img)
            
        plt.show()
        
    def histogram(imgs:np.ndarray|torch.Tensor, figsize:tuple=(6, 2), show_rgb:bool=False):
        
        if isinstance(imgs, torch.Tensor):
            imgs = View.chan_f2l(imgs.double().numpy())
            imgs = (imgs * 255).astype(np.uint8)
            
        if len(imgs.shape) == 3:
            imgs = View.add_dim(imgs)
        
        for i, img in enumerate(imgs):
            rgb_hist = np.zeros(shape=(3, 256))
            
            for j in range(3):
                for x in range(img.shape[0]):
                    for y in range(img.shape[1]):
                        rgb_hist[j, img[x, y, j]] += 1
                        
                        
            
            
            plt.figure(figsize=figsize)
            
            if show_rgb:
                plt.plot(rgb_hist[0], color='red')
                plt.plot(rgb_hist[1], color='green')
                plt.plot(rgb_hist[2], color='blue')
            
            #plot the average of the three channels
            plt.plot((rgb_hist[0] + rgb_hist[1] + rgb_hist[2])/3, color='black')
            
        plt.show()
        
            
        
        
def sample_imgs_list(x:DataLoader, samples:int|slice|list = slice(0,1)):
    dataset_size = x.dataset[0][0].size()
    
    if isinstance(samples, int):
        x_iter = iter(x)
        count = 0
        
        samples_sel = torch.empty(size=(samples, *dataset_size))
        
        for idx, (batch, labels) in enumerate(x_iter):
            for img in batch:
                samples_sel[count] = img
                count += 1
                if count >= samples:
                    return samples_sel
    elif isinstance(samples, slice):
        dataset = x.dataset

        step = samples.step
        if step is None:
            step = 1
            
        samples_sel = torch.empty(size=((samples.stop-samples.start)//step, *dataset_size))
        count = 0
        for idx in range(samples.start, samples.stop, step):
            samples_sel[count] = dataset[idx][0]
            count += 1
        
        return samples_sel
    
    elif isinstance(samples, list):
        dataset = x.dataset

        samples_sel = torch.empty(size=(len(samples), *dataset_size))
        count = 0
        for idx in samples:
            samples_sel[count] = dataset[idx][0]
            count += 1
        
        return samples_sel
    else:
        raise ValueError('samples must be of type int or slice')