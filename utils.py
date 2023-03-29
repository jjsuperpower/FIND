import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# set matplotlib to show images with black background
plt.style.use('dark_background')

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

    # @staticmethod
    # def compare_color(before, after):
        
    #     if isinstance(before, torch.Tensor):
    #         before = View._torch2np(before)
    #     if isinstance(after, torch.Tensor):
    #         after = View._torch2np(after)
            
    #     if len(before.shape) == 3:
    #         before = View._add_dim(before)
    #         after = View._add_dim(after)
            
    #     if before.shape != after.shape:
    #         raise ValueError('Before and after must have same shape')
        
    #     fig, axs = plt.subplots(2, before.shape[0])
        
    #     if before.shape[0] > 1:
    #         for i in range(before.shape[0]):
    #             axs[0, i].axis('off')
    #             axs[0, i].imshow(before[i])
                
    #         for i in range(after.shape[0]):
    #             axs[1, i].axis('off')
    #             axs[1, i].imshow(after[i])
                
    #         axs[0,0].set_title('Before')
    #         axs[1,0].set_title('After')
    #     else:
    #         axs[0].axis('off')
    #         axs[0].imshow(before[0])
    #         axs[1].axis('off')
    #         axs[1].imshow(after[0])
            
    #         axs[0].set_title('Before')
    #         axs[1].set_title('After')
            
    #     plt.show()

    @staticmethod
    def compare3_color(before, after, orig):
        if isinstance(before, torch.Tensor):
            before = View._torch2np(before)
        if isinstance(after, torch.Tensor):
            after = View._torch2np(after)
        if isinstance(orig, torch.Tensor):
            orig = View._torch2np(orig)
            
        if before.shape == after.shape == orig.shape:
            pass
        else:
            raise ValueError('Before and after must have same shape')
            
        if len(before.shape) == 3:
            before = View._add_dim(before)
            after = View._add_dim(after)
            orig = View._add_dim(orig)
        
        fig, axs = plt.subplots(3, before.shape[0])
        
        if before.shape[0] > 1:
            for i in range(before.shape[0]):
                axs[0, i].axis('off')
                axs[0, i].imshow(before[i])
                
            for i in range(after.shape[0]):
                axs[1, i].axis('off')
                axs[1, i].imshow(after[i])
                
            for i in range(orig.shape[0]):
                axs[2, i].axis('off')
                axs[2, i].imshow(orig[i])
                
            axs[0,0].set_title('Before')
            axs[1,0].set_title('After')
            axs[2,0].set_title('Original')
        else:
            axs[0].axis('off')
            axs[0].imshow(before[0])
            axs[1].axis('off')
            axs[1].imshow(after[0])
            axs[2].axis('off')
            axs[2].imshow(before[0])
            
            axs[0].set_title('Before')
            axs[1].set_title('After',)
            axs[2].set_title('Original')
            
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
    def compare_color(imgs_list:list[np.ndarray| torch.Tensor]| np.ndarray| torch.Tensor, labels:list[str]|str=None, figsize:tuple=(10, 16)) -> None:
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
        fig, axs = plt.subplots(len(imgs_list), len(imgs_list[0]), figsize=figsize)
            
        # if only one image group, plot each image in a column
        if len(imgs_list) == 1:
            for i, img in enumerate(imgs_list[0]):
                axs[i].axis('off')
                axs[i].set_title(labels[0])
                axs[i].imshow(img)

        # if only one image in each image group, plot each image group in a row
        elif len(imgs_list[0]) == 1:
            for i, imgs in enumerate(imgs_list):
                axs[i].axis('off')
                axs[i].set_title(labels[i])
                axs[i].imshow(imgs[0])
                
        else:
            for i, imgs in enumerate(imgs_list):        # rows
                for j, img in enumerate(imgs):          # columns
                    axs[i, j].axis('off')
                    axs[i, j].set_title(labels[i])
                    axs[i, j].imshow(img)
            
        plt.show()
        
        
def sample_imgs_list(x:DataLoader, samples:int|slice = slice(0,1)):
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
    else:
        raise ValueError('samples must be of type int or slice')