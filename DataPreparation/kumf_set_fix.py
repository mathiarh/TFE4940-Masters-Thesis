#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:26:46 2022

@author: mathiasrammhaugland
"""

import os
from xml.dom import minidom
import json
from PIL import Image
from matplotlib import pyplot as plt
import cv2

#after chosen images were put in a folder,
#this script sorts the images in two folders according to class,
#makes a json file with image/polyp characteristics from xml-files,
#and stores all images with gt box drawn around them in a folder

path = '/Users/mathiasrammhaugland/Documents/Masteroppgave/ThesisDatasets/set3_test_KUMC'



def get_xml_path(im_name):
    name_lst = im_name.split('_')
    set_ind = name_lst[0]
    xml_number = name_lst[-1].replace('jpg','xml')
    
    if set_ind != 'train':
        folder_ind = name_lst[1]
        spec_path = set_ind+"2019/"+"Annotation/"+folder_ind+"/"+xml_number
    else:
        spec_path = set_ind+"2019/"+"Annotation/"+xml_number
        

    xml_path = '/Users/mathiasrammhaugland/Documents/Masteroppgave/Datasett/PolypsSet/'+spec_path
    return xml_path
    
    
def get_characteristics_dic(xml_path):
    doc = minidom.parse(xml_path)
    im_w = doc.getElementsByTagName('width')
    im_h = doc.getElementsByTagName('height')
    label = doc.getElementsByTagName('name')
    xmin = doc.getElementsByTagName('xmin')
    ymin = doc.getElementsByTagName('ymin')
    xmax = doc.getElementsByTagName('xmax')
    ymax = doc.getElementsByTagName('ymax')
    
    
    dic = {}
    dic['height'] = im_h[0].childNodes[0].data
    dic['width'] = im_w[0].childNodes[0].data
    box = []
    box_dic = {}
    box_dic["xmin"] = xmin[0].childNodes[0].data
    box_dic["ymin"] = ymin[0].childNodes[0].data
    box_dic["xmax"] = xmax[0].childNodes[0].data
    box_dic["ymax"] = ymax[0].childNodes[0].data
    
    if label[0].childNodes[0].data=='adenomatous':
        box_dic["label"] = 'Adenoma'
    elif label[0].childNodes[0].data=='hyperplastic':
        box_dic["label"] = 'Hyperplasia'
    else: print("PROBLEM")
    
    
    box.append(box_dic)
    dic["bbox"] = box   
    
    return dic


 


def add_dics_to_json(dic_complete):   
    with open(path+"/set3.json",'w') as fil:
        json.dump(dic_complete,fil)
    

def store_from_class(dic,name):
    for key, val in dic.items(): #only one it
        if key== 'bbox':
            clas = val[0]['label']
    im = Image.open(f"{path}/images/{name}")
    im.save(f"{path}/{clas}/{name}")
        
    #open image from im_path,
    #save in new folder based on class

def store_im_box(dic,name):
    #print(dic)
    for key, val in dic.items():
        if key == 'bbox':
            box =  val[0]
    xmin = int(box["xmin"])
    ymin = int(box["ymin"])
    xmax = int(box["xmax"])
    ymax = int(box["ymax"])
        
    label = box["label"]
        
        
        
    imgcv = cv2.imread(f"{path}/{label}/{name}")
    color = (0,255,0)
    results = [
    {
     "left": xmin,
     "top": ymin,
     "width": xmax-xmin,
     "height": ymax-ymin,
     "label": label
     }]
    for res in results:
        left = int(res['left'])
        top = int(res['top'])
        right = int(res['left']) + int(res['width'])
        bottom = int(res['top']) + int(res['height'])
        label = res['label']
        imgHeight, imgWidth, _ = imgcv.shape
        thick = int((imgHeight + imgWidth) // 900)
        cv2.rectangle(imgcv,(left, top), (right, bottom), color, thick)
        cv2.putText(imgcv, label, (left, top - 12), 0, 1e-3 * imgHeight, color, thick//3)
    cv2.imwrite(path+"/all_images_with_boxes/"+name, imgcv)


def main():
    #iterate through set3_test_KUMC
    dic_complete = {}
    #a,h = 0,0
    for im_name in os.listdir(f"{path}/images"):
    
        #find xml file path for image:
        xml_path = get_xml_path(im_name)

        #open it and get class, heigh, width, bbox, etc:
        dic = get_characteristics_dic(xml_path)
        
        """
        print(im_name+" " + dic['bbox'][0]['label'])
        if dic['bbox'][0]['label'] == 'Adenoma':
            a = a+1
        elif dic['bbox'][0]['label'] == 'Hyperplasia':
            h = h+1
        else: print("HOUSTON")
        """
        
        dic_complete[im_name] = dic
        
        #store image in adenoma or hp folder:
        store_from_class(dic,im_name)
        
                
        #input dic file and print image with box:
        store_im_box(dic,im_name)
    
    #make json_file:
    add_dics_to_json(dic_complete)

    #print(f"A:{a} and H:{h}")

    
if __name__ == '__main__':
    main()
