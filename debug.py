import torch
import torchvision
import torchvision.models as models
from torchvision import transforms
from torchvision.models.resnet import ResNet50_Weights
import lightning.pytorch as pl
from ultralytics import YOLO

from mymodels import Model_Wrapper, Preprocess
from myutils import View, sample_imgs_list, compare_ds

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

PATH_TO_COCO = '/f_storage/datasets/coco/2017'
NUM_IMG_EVAL = 10000
prep = Preprocess(PATH_TO_COCO, (640, 640), dataset_type='coco', shuffle=False, batch_size=16, num_workers=1)
preview_img_slice = slice(00000, 50000, 10000)

from yolo_wrapper import YOLOv8_COCO_Wrapper
from coco_ds import CocoDataset, CocoResults

prep.reset_trans()
coco = prep.get_loader()
model = YOLOv8_COCO_Wrapper(CocoResults(coco.dataset), 'yolov8s.pt')
model = Model_Wrapper(model, model_type='yolo')
trainer = pl.Trainer(accelerator='auto', limit_test_batches=400)

trainer.test(model, coco)