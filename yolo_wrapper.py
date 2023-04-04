import torch
from ultralytics import YOLO
from copy import deepcopy
import numpy as np
import contextlib
from ultralytics.yolo.utils.ops import coco80_to_coco91_class

from coco_ds import CocoResults
from myutils import Void


COCO_80_TO_91 = coco80_to_coco91_class()

class YOLOv8_COCO_Wrapper(YOLO):
    ''' Wrapper for YOLOv8 and COCO Dataset
    This used to add a layer of work arounds for using Tensors instead of PIL images.
    At the time of writing (April 2023), Yolov8 is new and has some bugs.
    If you are using this in the future, you may not need this class.'''
    
    def __init__(self, coco_evaluator:CocoResults, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coco_evaluator = coco_evaluator
        
    def reset_coco_evaluator(self):
        self.coco_evaluator.reset()
        return self
        
        
    def predict_coco(self, img_ids:int|list, imgs:torch.Tensor, orig_size:list[tuple[int,int]], gen_plot:bool=False):
        
        if isinstance(img_ids, int):
            img_ids = [img_ids]
            
        if isinstance(orig_size, tuple):
            orig_size = [orig_size]
        
        img_size = imgs.size()[-2:]
        results = self.predict(imgs*255)
        
        for idx, result in enumerate(results):
            result = deepcopy(result)
            coco_pred = []
            boxes = result.boxes
            
            # print(boxes)
            
            for i in range(len(boxes.cls)):
                label = COCO_80_TO_91[int(boxes.cls[i].cpu().item())]
                bbox = boxes.xyxy[i].clone()
                bbox[0] *= orig_size[idx][0] / img_size[0]       # rescale to original image size
                bbox[1] *= orig_size[idx][1] / img_size[1]
                bbox[2] *= orig_size[idx][0] / img_size[0]
                bbox[3] *= orig_size[idx][1] / img_size[1]
                bbox[2] = bbox[2] - bbox[0]                 # convert to xywh
                bbox[3] = bbox[3] - bbox[1]
                bbox = bbox.round().int()
                bbox = bbox.cpu().tolist()
                
                score = boxes.conf[i].cpu().item()
                coco_pred.append((label, bbox, score))
                
            self.coco_evaluator.add_results(coco_pred, img_ids[idx].cpu().int().item())
            
        if gen_plot:
            plots = self.plot(imgs, results)
        else:
            plots = None
        
        return self.coco_evaluator, plots
    
    
    
    def plot(self, imgs:torch.Tensor, results:list[dict]=None):
        
        if results is None:
            results = self.predict(imgs*255)
            
        plots = torch.empty_like(imgs)
            
        for idx, result in enumerate(results):
        
            img_to_plot = imgs[idx].clone() * 255

            img_to_plot = img_to_plot.permute(1,2,0)
            img_to_plot = img_to_plot.cpu().numpy()
            img_to_plot = np.ascontiguousarray(img_to_plot)
            result.orig_img = img_to_plot
            plot = result.plot()
            plot = torch.Tensor(plot)
            plot = plot.permute(2,0,1)
            plot = plot / 255
            plots[idx] = plot
        
        return plots
    
    def __call__(self, *args, **kwargs):
        return self.predict_coco(*args, **kwargs)
            
        
        
        
        