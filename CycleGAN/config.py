#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:48:26 2021
Modified on Fri Mar 04 13:26:62

@author: mathiasrammhaugland

from:
https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/CycleGAN
"""

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/home/hemin/mathiasrammhaugland/master/CycleGAN/train_data/OUS-NBI-ColonVDB-r"
#VAL_DIR = "/content/drive/My Drive/CycleGAN/data/val"
TEST_DIR = "/home/hemin/mathiasrammhaugland/master/set3/WLI/Adenoma"
#from paper:
BATCH_SIZE = 1 
LEARNING_RATE = 2e-4 #2e-5? 1e-5?
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 60
LOAD_MODEL = True
SAVE_MODEL = True

#load checkpoints
LOAD_CHECKPOINT_GEN_H = "/home/hemin/mathiasrammhaugland/master/CycleGAN/checkpoints/run3_ous/genh_512.pth.tar"
LOAD_CHECKPOINT_GEN_Z = "/home/hemin/mathiasrammhaugland/master/CycleGAN/checkpoints/run3_ous/genz_512.pth.tar"
LOAD_CHECKPOINT_CRITIC_H = "/home/hemin/mathiasrammhaugland/master/CycleGAN/checkpoints/run3_ous/disc_512.pth.tar"
LOAD_CHECKPOINT_CRITIC_Z = "/home/hemin/mathiasrammhaugland/master/CycleGAN/checkpoints/run3_ous/discz_512.pth.tar"

#save checkpoints
CHECKPOINT_GEN_H = "/home/hemin/mathiasrammhaugland/master/CycleGAN/checkpoints/run5_ous/genh_512.pth.tar"
CHECKPOINT_GEN_Z = "/home/hemin/mathiasrammhaugland/master/CycleGAN/checkpoints/run5_ous/genz_512.pth.tar"
CHECKPOINT_CRITIC_H = "/home/hemin/mathiasrammhaugland/master/CycleGAN/checkpoints/run5_ous/disch_512.pth.tar"
CHECKPOINT_CRITIC_Z = "/home/hemin/mathiasrammhaugland/master/CycleGAN/checkpoints/run5_ous/discz_512.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=512, height=512), #originaly 256x256
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit = 180, p=0.5),
        A.Transpose(p=0.5),
        A.Affine(scale=(0.8,1.2), p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255), # Hemin: we can also compute these values from our training dataset
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
