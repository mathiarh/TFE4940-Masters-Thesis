#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import torch
from dataset import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
import os
import numpy as np

def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, scheduler_disc, scheduler_gen):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss)/2

        opt_disc.zero_grad()

        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)

        

        d_scaler.update()
        


        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
#            identity_zebra = gen_Z(zebra)
#            identity_horse = gen_H(horse)
#            identity_zebra_loss = l1(zebra, identity_zebra)
#            identity_horse_loss = l1(horse, identity_horse)

            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_zebra_loss * config.LAMBDA_CYCLE
                + cycle_horse_loss * config.LAMBDA_CYCLE
#                + identity_horse_loss * config.LAMBDA_IDENTITY
#                + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()

        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        

        
        g_scaler.update()
        

        #save every 200. image during training
        if idx % 200 == 0:
            save_image(horse*0.5+0.5, f"/home/hemin/mathiasrammhaugland/master/CycleGAN/saved_images/fromWLI_5/wli_{idx}.png")
            save_image(fake_zebra*0.5+0.5, f"/home/hemin/mathiasrammhaugland/master/CycleGAN/saved_images/fromWLI_5/wli_nbi_{idx}.png")
            save_image(cycle_horse*0.5+0.5, f"/home/hemin/mathiasrammhaugland/master/CycleGAN/saved_images/fromWLI_5/wli_nbi_wli_{idx}.png")
            save_image(zebra*0.5+0.5, f"/home/hemin/mathiasrammhaugland/master/CycleGAN/saved_images/fromNBI_5/nbi_{idx}.png")
            save_image(fake_horse*0.5+0.5, f"/home/hemin/mathiasrammhaugland/master/CycleGAN/saved_images/fromNBI_5/nbi_wli_{idx}.png")
            save_image(cycle_zebra*0.5+0.5, f"/home/hemin/mathiasrammhaugland/master/CycleGAN/saved_images/fromNBI_5/nbi_wli_nbi_{idx}.png")
        

        loop.set_postfix(H_real=H_reals/(idx+1), H_fake=H_fakes/(idx+1))



def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    
    #for scheduling learning rate decay
    
    #only linear decay
    #lambda1 = lambda epoch: max(0,1.0 - ((epoch+1)*2e-6)/config.LEARNING_RATE)
    
    #constant followed by linear decay
    lr_lst = np.ones(1)
    for i in range(1,61,1):
    	lamb = max(0,1.0 - (i/60))
    	print(lamb)
    	lr_lst = np.append(lr_lst,lamb)
    print(lr_lst)
    lambda1 = lambda epoch: lr_lst[epoch]
    
    
    
    scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda = lambda1)
    scheduler_gen = torch.optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda = lambda1)

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.LOAD_CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.LOAD_CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.LOAD_CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.LOAD_CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE,
        )
    

    # New function for creating dataset
    dataset = 0
    for folder in os.listdir(config.TRAIN_DIR+"/WLI"):
    	batch = HorseZebraDataset(
    	    root_horse=config.TRAIN_DIR+"/WLI/" + folder, root_zebra=config.TRAIN_DIR+"/NBI/"+folder, transform=config.transforms)
    	if (dataset == 0):
    	    dataset = batch
    	else:
    	    dataset = torch.utils.data.ConcatDataset([dataset,batch])
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS,pin_memory=True)
 
    	
    	
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        print("EPOCH {}/{}".format(epoch+1,config.NUM_EPOCHS))
        print("LR: "+str(lr_lst[epoch]))
        train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, scheduler_disc, scheduler_gen)

            
        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)
    
    
            #saving independent checkpoints every 5th epoch as well:
            if (epoch % 5 == 0):
                save_checkpoint(gen_H, opt_gen, filename = f"/home/hemin/mathiasrammhaugland/master/CycleGAN/checkpoints/run5_ous/genh_512_epoch_{epoch}.pth.tar")
                save_checkpoint(gen_Z, opt_gen, filename = f"/home/hemin/mathiasrammhaugland/master/CycleGAN/checkpoints/run5_ous/genz_512_epoch_{epoch}.pth.tar")
                save_checkpoint(disc_H, opt_disc, filename = f"/home/hemin/mathiasrammhaugland/master/CycleGAN/checkpoints/run5_ous/disch_512_epoch_{epoch}.pth.tar")
                save_checkpoint(disc_Z, opt_disc, filename = f"/home/hemin/mathiasrammhaugland/master/CycleGAN/checkpoints/run5_ous/discz_512_epoch_{epoch}.pth.tar")
         
        d_scale = d_scaler.get_scale() #from me

        #from me
        d_skip_lr_sched = (d_scale > d_scaler.get_scale())
        if not d_skip_lr_sched:
        	scheduler_disc.step()
        g_scale = g_scaler.get_scale() #from me
        #from me
        g_skip_lr_sched = (g_scale < g_scaler.get_scale())
        if not g_skip_lr_sched:
        	scheduler_gen.step()
    

if __name__ == "__main__":
    main()
    
    
    
    
    
