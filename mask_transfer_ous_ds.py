#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 09:01:45 2022

@author: mathiasrammhaugland
"""

import os
import cv2

#IDEA:
    #Go through each video+TYPE folder of images in the cleaned set
    #Put all masks in this folder's names into an array
        #This is (also) the names of the masks we want to transfer
    #Then iterate through the corresponding uncleaned mask folder
    #If its name is similar to a name in the array (the image it corresponds to),
    #then copy it over to the new dataset


def move_masks():
    path_old = "/media/hemin/Data/Hemin'sFiles/My_Dataset/NBI_White_Video/GT" #??, old folder containing all masks
    path_new = "/media/hemin/Data/Hemin'sFiles/My_Dataset/OUS-NBI-ColonVDB-reduced/"
    #add: path_new = "media/hemin/Data/My_Dataset/OUS-NBI-ColonVDB/data"
    #new folder with only masks that correspond to selected images in the cleaned dataset, should point both to (new) masks and imgs
    
    #Into old path
    for i in range(24):
        for TYPE in (["NBI/", "RGB/"]):
            
            #add new img names in an array
            new_img_names = []
            for image_name in os.listdir(path_new + "data/" + TYPE + "v{}/".format(i+1)): #in specific video and TYPE folders
                
                #remove file type
                new_img_names.append(os.path.splitext(image_name)[0]) #add image file name to array
            
            #get into the same video and TYPE folder that the masks will be cloned from
            for mask in os.listdir(path_old + "/Video_{}/".format(i+1) + TYPE):
                
                #check if mask exists in reduced/new image dataset array
                mask_r = os.path.splitext(mask)[0] #remove file type
                if mask_r in new_img_names: #check that file name actually should be equal, they might have been changed!!!   
                    
                    #then copy the mask to new mask folder
                    mask_img = cv2.imread(path_old + "/Video_{}/".format(i+1) + TYPE + mask) #read the image that will be copied
                    path_new_mask = os.path.join(path_new + "GT/" + TYPE + "v{}/".format(i+1) + mask) #define new path
                    #maybe rename new mask
                    cv2.imwrite(path_new_mask, mask_img)
                    
            
if __name__ == '__main__':
    move_masks()