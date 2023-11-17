from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch
import contextlib
from copy import deepcopy
import json

from myutils import Void

COCO_classes = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe",
    "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", 
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", 
    "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk",
    "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
    "toothbrush", "hair brush"
]
        

class CocoDataset(Dataset):
    def __init__(self, root: str, set: str, size: int=None, transform=None, quiet: bool=True):
        super().__init__()
        self.transform = transform
        self.rootPath = Path(root)
        self.dataType = set
        self.quiet = quiet
        annPath = self.rootPath.joinpath('annotations', f'instances_{set}.json')
        if quiet:
            with contextlib.redirect_stdout(Void):
                self.coco = COCO(annPath)
        else:
            self.coco = COCO(annPath)
        
        self.imgIds = self.coco.getImgIds()
        if size is not None:
            self.imgIds = self.imgIds[:size]
        self.imgIds = sorted(self.imgIds)
            
    def get_by_id(self, imgId: int):
        img_obj = self.coco.loadImgs(imgId)[0]
        img_path = self.rootPath.joinpath(self.dataType, img_obj['file_name'])
        img = Image.open(img_path)
        return img

    def __getitem__(self, index: int):
        imgId = self.imgIds[index]
        img = self.get_by_id(imgId)
        img_size = img.size
        
        if img.mode != 'RGB':
            if not self.quiet:
                print(f'Warning: grayscale image detected (ID = {imgId}), converting to RGB')
            new_img = Image.new('RGB', img.size)
            new_img.paste(img)
            img = new_img
            
        if self.transform is not None:
            img = self.transform(img)
        
        return imgId, img, torch.Tensor([*img_size])
    
    def __len__(self):
        return len(self.imgIds)
    
    
class CocoResults:
    def __init__(self, coco_dataset: CocoDataset, verbose:bool=False) -> None:
        self.coco_dataset = coco_dataset
        self.verbose = verbose
        self.coco_results = []
        self.img_ids = []

    def add_results(self, results: list, image_id: int):
        for label, bbox, score in results:
           
            self.coco_results.append({
                "image_id": image_id,
                "category_id": label,
                "bbox": [p for p in bbox],
                "score": score
            })
        self.img_ids.append(image_id)
        return self
    
    def reset(self):
        self.coco_results = []
        self.img_ids = []
        return self
            
    def get_results(self):
        return deepcopy(self.coco_results)
            
    def save_results(self, filename:str):
            
        with open(filename, 'w') as f:
            json.dump(self.coco_results, f, indent=4)
            
        return self
            
    def load_results(self):
        with open(self.file_name, 'r') as f:
            self.coco_results = json.load(f)
        self.img_ids = list(set([r['image_id'] for r in self.coco_results]))
        
        return self
    
    def get_avg_conf(self):
        return sum([r['score'] for r in self.coco_results]) / len(self.coco_results)
        
    def _eval(self, iouType: str='bbox'):
        
        coco = self.coco_dataset.coco
        cocoValPred = coco.loadRes(self.coco_results)
        coco_eval = COCOeval(coco, cocoValPred, iouType)
        
        if len(self.img_ids) > 0:
            coco_eval.params.imgIds = self.img_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
            
        result = {}
        
        result['AP'] = coco_eval.stats[0]
        result['AP50'] = coco_eval.stats[1]
        result['AP75'] = coco_eval.stats[2]
        result['APs'] = coco_eval.stats[3]
        result['APm'] = coco_eval.stats[4]
        result['APl'] = coco_eval.stats[5]
        result['AR1'] = coco_eval.stats[6]
        result['AR10'] = coco_eval.stats[7]
        result['AR100'] = coco_eval.stats[8]
        result['ARs'] = coco_eval.stats[9]
        result['ARm'] = coco_eval.stats[10]
        result['ARl'] = coco_eval.stats[11]
        result['conf'] = self.get_avg_conf()
        
        return result

    def eval(self, *args, **kwargs):
        if self.verbose:
            return self._eval(*args, **kwargs)
        else:
            with contextlib.redirect_stdout(Void):
                return self._eval(*args, **kwargs)

    
        
if __name__ == '__main__':
    
    from matplotlib import pyplot as plt
    import numpy as np
    
    TEST_DS_PATH = '../../datasets/coco/2017/'
    
    dataset = CocoDataset(TEST_DS_PATH, 'val2017')
    
    print(f'Number of images: {len(dataset)}')
    
    img_id, img, img_size = dataset[1]
    
    print(f'Image id: {img_id}')
    print(f'Image size: {img_size}')
    
    np_img = np.array(img)
    plt.imshow(np_img)
    plt.show()
    
    # test image for mAP evaluation
    img = dataset.get_by_id(285)
    
    label, bbox, score  = 23, [0.0, 50, 600.0, 600.0],  0.9
    fake_results = CocoResults(dataset).add_results([(label, bbox, score)], 285)
    eval = fake_results.eval()
    
    print(eval['AP'])
    
    
