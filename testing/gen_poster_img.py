import numpy as np
from PIL import Image
from torchvision import transforms as tf

from transforms import Luminance, Fog, Rain, Hist_EQ, Retinex

SAVE_PATH = 'images/poster_imgs/'
IMG_FILENAME = 'images/ILSVRC2012_val_00033521.JPEG'
img_orig = Image.open(IMG_FILENAME)

img_orig_tensor = tf.ToTensor()(tf.Resize((350, 650))(img_orig))

img_effects = [Luminance(1/8), Luminance(2), Fog(10), Rain(2, 75)]
img_effects_names = ['dark', 'overexposed', 'foggy', 'darkrainy']

img_enhance = [Hist_EQ(), Retinex('SSR', 400)]
img_enhance_names = ['histeq', 'retinex']


# Save original image
tf.ToPILImage()(img_orig_tensor).save(SAVE_PATH + 'original.png')

for i, effect in enumerate(img_effects):
    img_augmented = effect(img_orig_tensor)
    img_pil = tf.ToPILImage()(img_augmented)
    img_pil.save(SAVE_PATH + f'{img_effects_names[i]}.png')
    
    for j, enh in enumerate(img_enhance):
        img_enh = enh(img_augmented)
        img_pil = tf.ToPILImage()(img_enh)
        img_pil.save(SAVE_PATH + f'{img_effects_names[i]}_{img_enhance_names[j]}.png')
