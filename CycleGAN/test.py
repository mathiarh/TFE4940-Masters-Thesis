#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:52:57 2021

@author: mathiasrammhaugland
"""
import torch
from dataset import OnlyHorseDataset
from utils import load_checkpoint
from torch.utils.data import DataLoader
import torch.optim as optim
import config
from torchvision.utils import save_image
from torchvision.transforms import functional as F
from discriminator_model import Discriminator
from generator_model import Generator
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
import time

import os
from PIL import Image

test_transforms = A.Compose(
    [
        A.Resize(width=512, height=512), #originaly 256x256
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    #additional_targets={"image"},
)

def test():
    #disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    #disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    
    """
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    """

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )


    load_checkpoint(
        config.LOAD_CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
    )
    load_checkpoint(
        config.LOAD_CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
    )
    """
    load_checkpoint(
        config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE,
    )
    load_checkpoint(
        config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE,
    )
    """
    
    #Function for storing orginal image sizes
    #assumes test images are in testdir --> WLI and NBI
    #saves image width and height in dictionary {WLI: {im1: (width,height), im2: (width, height),...}, NBI: {im1: (width,height),...}}
    height_width_dic = {}
    for im_name in os.listdir(config.TEST_DIR):
        im = Image.open(config.TEST_DIR+"/"+im_name)
        height_width_dic[im_name] = im.size
    
    
    """
    test_folders = [4,6,8,10,14,16,18,20,22,24]
    for i in test_folders:
        batch = HorseZebraDataset(
            root_horse = config.TEST_DIR+"/RGB/v{}".format(i), root_zebra=config.TEST_DIR+"/NBI/v{}".format(i), transform = test_transforms
            )
        if i == test_folders[0]:
            testset = batch
        else:
            testset = torch.utils.data.ConcatDataset([testset,batch])
    """
    testset = OnlyHorseDataset(root_horse=config.TEST_DIR, transform=test_transforms)

    testloader = DataLoader(
        testset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    loop = tqdm(testloader, leave=True)
    
    time_start = time.time()
    t_save_all = 0
    
    for idx, (horse,horse_name) in enumerate(loop):
        horse = horse.to(config.DEVICE)
        horse_name=horse_name[0].split(".")[0]
        
        # Test
        with torch.cuda.amp.autocast():
            #fake_horse = gen_H(zebra)
            #cycle_zebra = gen_Z(fake_horse)
            
            fake_zebra = gen_Z(horse)
            #cycle_horse = gen_H(fake_zebra)

        #Resize function, by using height_width_dic
        fzebra_w,fzebra_h = height_width_dic[horse_name+".jpg"]
        fzebra_hw = tuple([fzebra_h,fzebra_w])
        fake_zebra_resized = F.resize(fake_zebra,fzebra_hw)
        
        t_save = time.time()
        save_image(fake_zebra_resized*0.5+0.5, f"/home/hemin/mathiasrammhaugland/master/set3/SNBI2/Adenoma/{horse_name}.jpg")
        t_save_all = t_save_all + time.time()-t_save
    	
    time_stop = time.time()
    print(f"Iterations: {idx+1}")
    it_time = (time_stop - time_start)/(idx+1)
    print(f"Run time: {it_time}")
    print(f"Run time without saving time: {it_time - t_save_all/(idx+1)}")
            
if __name__ == "__main__":
    test()
