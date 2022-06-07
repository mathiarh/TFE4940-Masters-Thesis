#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:22:45 2022

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
    path_old = "/Users/mathiasrammhaugland/Documents/Masteroppgave/ThesisDatasets/set2_onlyPICCOLO/train/masks_old" #??, old folder containing all masks
    path_new = "/Users/mathiasrammhaugland/Documents/Masteroppgave/ThesisDatasets/set2_onlyPICCOLO/train"
    #add: path_new = "media/hemin/Data/My_Dataset/OUS-NBI-ColonVDB/data"
    #new folder with only masks that correspond to selected images in the cleaned dataset, should point both to (new) masks and imgs
    
    #Into old path
    for i in range(1):
        #for TYPE in (["NBI/", "WLI/"]):
        for i in range(1): #alt to above
            
            #add new img names in an array
            new_img_names = []
            #for image_name in os.listdir(path_new + TYPE + "polyps/"): #in specific video and TYPE folders
            for image_name in os.listdir(path_new + "/polyps"): #alt to above
                
                #remove file type
                new_img_names.append(os.path.splitext(image_name)[0]) #add image file name to array

            #get into the same video and TYPE folder that the masks will be cloned from
            for mask in os.listdir(path_old):

                #check if mask exists in reduced/new image dataset array
                mask_r = os.path.splitext(mask)[0] #remove file type
                mask_r2 = mask_r.replace('_Corrected','')

                if mask_r2 in new_img_names: #check that file name actually should be equal, they might have been changed!!!   
                    
                    #then copy the mask to new mask folder
                    mask_img = cv2.imread(path_old + "/" + mask) #read the image that will be copied
                    #path_new_mask = os.path.join(path_new + TYPE + "masks/" + mask_r2 + ".tif") #define new path
                    path_new_mask = os.path.join(path_new+"/masks/"+mask_r2+".tif") #alt to above
                    #maybe rename new mask
                    cv2.imwrite(path_new_mask, mask_img)
                    
            
if __name__ == '__main__':
    move_masks()
