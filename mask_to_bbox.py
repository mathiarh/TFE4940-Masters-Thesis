#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:34:38 2022

@author: mathiasrammhaugland
"""

from PIL import Image
import numpy as np
import cv2
from skimage.measure import regionprops, label
from matplotlib import pyplot as plt
import os
import json
from pandas import *

path = "/Users/mathiasrammhaugland/Documents/Masteroppgave/ThesisDatasets/set2_onlyPICCOLO/NBI_ref/validation/"

def loadIm(path, mode= "RGB"):
    im = Image.open(path).convert(mode)
    return np.asarray(im)

def displayArrayIm(im, mode="RGB"):
    im = Image.fromarray(im, mode)
    im.show()
    return None

def binarizeMask(arr):
    th, arr_th = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY)
    return arr_th


def addToDict(im, props, label):
    dic = {}
    dic['height'] = im.shape[0]
    dic['width'] = im.shape[1]
    box_arr = []

    for prop in props:
        #make box_dic
        box_dic = {}
        #box_dic["label"] = "polyp" #for only detection
        box_dic["label"] = label #for detection and classification
        box_dic["xmin"] = prop.bbox[1]
        box_dic["ymin"] = prop.bbox[0]
        box_dic["xmax"] = prop.bbox[3]
        box_dic["ymax"] = prop.bbox[2]
        
        #add box_dic to box_arr only if area is over 300 pixels
        if ((box_dic["xmax"]-box_dic["xmin"])*(box_dic["ymax"]-box_dic["ymin"])) > 300:
            box_arr.append(box_dic)
    
    #add box_arr to dic
    dic["bbox"] = box_arr
    return dic


def save_dict(dic, name):
    with open(name+'.json', 'w') as outfile:
        json.dump(dic, outfile)
    
        

def dispImWithBBoxes(im, dic):
    for box in dic["bbox"]:        
        start_p = (box["xmin"], box["ymin"])
        end_p = (box["xmax"], box["ymax"])
        im = cv2.rectangle(im, start_p, end_p, (0,255,0),2)
    displayArrayIm(im, "RGB")
    
    
def getLabels(): #For classification
    #Reading labels from .csv file
    csv_file_path = '/Users/mathiasrammhaugland/Documents/Masteroppgave/ThesisDatasets/set2_onlyPICCOLO/clinical metadata_release0.1.csv'
    data = read_csv(csv_file_path, sep=';', encoding_errors = 'ignore')
    
    polyp_names = data["CODE - LESION"].tolist()
    polyp_types = data["LITERAL DIAGNOSIS (Pathologist)"].tolist()
  
    class_labels = dict(zip(polyp_names,polyp_types))
    return class_labels
    
def test(img_path, image, json_file):
    #Show image with bbox acquired from bbox_dict
    im_arr = loadIm(img_path+image, "RGB") #load image as array
    with open(json_file+'.json') as file: #open json file
        bbox_dict = json.load(file)
    dispImWithBBoxes(im_arr, bbox_dict[image]) #display correct box(es) on chosen image


def main():

    bbox_dict = {}
    mask_path = path + 'masks/'
    img_path = path + 'polyps/'
    bbox_file = path + 'eval_piccolo_nbi' #what to name the JSON file
    class_labels = getLabels() #when classification, get a dict of polyp names with corresponding classification. #TRAIN with class
    #class_labels = dict(zip(['v6','v17','v21','v22'],["Adenoma","Hyperplasia","Adenoma","Hyperplasia"])) #EVAL detection and classification

    for filename in os.listdir(mask_path):

        mask_arr = loadIm(mask_path+filename, 'L') #Load mask as array
        mask_th = binarizeMask(mask_arr) #binarize
        
        im_filename = filename.replace('.tif','.png') #if mask and image have different filetypes (mask,image)
        im_arr = loadIm(img_path+im_filename, 'RGB') #Read corresponding image
   
        label_masks = label(mask_th)
        
        props = regionprops(label_masks)
        
        polyp_no = im_filename.split('_')[0] #get polyp name/number
        class_label = class_labels[int(polyp_no)] #get label of this specific polyp from the dict TRAIN
        #class_label = class_labels[polyp_no] #EVAL detection and classification
        #class_label="polyp" #only detection
        bbox_dict[im_filename] = addToDict(im_arr, props, class_label) #Put BBox in bbox_dict


    save_dict(bbox_dict,bbox_file)
    
    
    #test(img_path, "002_VP2_frame1024.png", bbox_file) #opens json file and adds boxes to a chosen image
    
    
    
if __name__ == '__main__':
    main()
    