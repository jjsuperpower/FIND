# Setting Up Datasets

## Required Datasets
- ImageNet (ILSVRC2012)
- COCO 2017

### ImageNet
Download the ImageNet validation dataset and devkit from [here](https://image-net.org/download.php) (You will need to make an account). There should be no need to manually extract the files as Torch Vision should do it automatically.

### COCO 2017
Download the COCO 2017 validation dataset and annotations from [here](http://images.cocodataset.org/zips/val2017.zip) and [here](http://images.cocodataset.org/annotations/annotations_trainval2017.zip). Extract the downloaded into the directory structure show below.

## Example Directory Structure
```
📂datasets
 ┣ 📂coco
 ┃ ┣ 📂2017
 ┃ ┃ ┣ 📂annotations
 ┃ ┃ ┃ ┗ 📜instances_val2017.json
 ┃ ┃ ┣ 📂labels
 ┃ ┃ ┃ ┣ 📂train2017
 ┃ ┃ ┃ ┃ ┣ 📜000000000009.txt
 ┃ ┃ ┃ ┃ ┣ 📜000000000025.txt
 ┃ ┃ ┃ ┃ ┣ 📜000000000030.txt
 ┃ ┃ ┃ ┃ ┣ ... A lot more images
 ┃ ┃ ┣ 📂test2017
 ┃ ┃ ┃ ┣ 📜000000000001.jpg
 ┃ ┃ ┃ ┣ 📜000000000016.jpg
 ┃ ┃ ┃ ┣ 📜000000000019.jpg
 ┃ ┃ ┃ ┣ ... A lot more images
 ┗ 📂imagenet
 ┃ ┗ 📂2012
 ┃ ┃ ┣ 📂ILSVRC2012_devkit_t12
 ┃ ┃ ┣ 📂val
 ┃ ┃ ┃ ┣ 📂n01440764
 ┃ ┃ ┃ ┣ 📂n01443537
 ┃ ┃ ┃ ┣ 📂n01484850
 ┃ ┃ ┃ ┣ ...
 ┃ ┃ ┣ 📜ILSVRC2012_devkit_t12.tar.gz
 ┃ ┃ ┣ 📜ILSVRC2012_img_train.tar       - Torch Vision will use this directly without extracting
 ┃ ┃ ┣ 📜ILSVRC2012_img_val.tar
 ┃ ┃ ┣ 📜meta.bin
```
> The training datasets are not required by this repo, but are included in example for completeness.
