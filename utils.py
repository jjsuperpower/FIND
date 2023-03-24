import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# set matplotlib to show images with black background
plt.style.use('dark_background')

class View:
    @staticmethod
    def _torch2np(tensor:torch.Tensor):
        array = tensor.numpy()
        if len(array.shape) == 4:
            array = array.swapaxes(1, 3).swapaxes(1, 2)
        elif len(array.shape) == 3:
            array = array.swapaxes(0, 2).swapaxes(0, 1)      # make channels last

        return array
    
    @staticmethod
    def _add_dim(array:np.ndarray):
        return array.reshape((1, *array.shape))

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
    def compare_color(before, after):
        
        if isinstance(before, torch.Tensor):
            before = View._torch2np(before)
        if isinstance(after, torch.Tensor):
            after = View._torch2np(after)
            
        if len(before.shape) == 3:
            before = View._add_dim(before)
            after = View._add_dim(after)
            
        if before.shape != after.shape:
            raise ValueError('Before and after must have same shape')
        
        fig, axs = plt.subplots(2, before.shape[0])
        
        if before.shape[0] > 1:
            for i in range(before.shape[0]):
                axs[0, i].axis('off')
                axs[0, i].imshow(before[i])
                
            for i in range(after.shape[0]):
                axs[1, i].axis('off')
                axs[1, i].imshow(after[i])
                
            axs[0,0].set_title('Before')
            axs[1,0].set_title('After')
        else:
            axs[0].axis('off')
            axs[0].imshow(before[0])
            axs[1].axis('off')
            axs[1].imshow(after[0])
            
            axs[0].set_title('Before')
            axs[1].set_title('After')
            
        plt.show()

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
        
        # fig, axs = plt.subplots(3, before.shape[0], figsize=(15,10))

        # for i in range(before.shape[0]):
        #     axs[0, i].axis('off')
        #     axs[0, i].imshow(before[i])
            
        # for i in range(after.shape[0]):
        #     axs[1, i].axis('off')
        #     axs[1, i].imshow(after[i])

        # for i in range(orig.shape[0]):
        #     axs[2, i].axis('off')
        #     axs[2, i].imshow(orig[i])
            
        # axs[0,0].set_title('Before', fontsize=16)
        # axs[1,0].set_title('After', fontsize=16)
        # axs[2,0].set_title('Original', fontsize=16)
        # plt.show()
        
def sample_imgs(x:DataLoader, samples:int|slice = slice(0,1)):
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