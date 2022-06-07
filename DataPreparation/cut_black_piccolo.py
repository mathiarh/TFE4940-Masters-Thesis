#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:37:32 2022

@author: mathiasrammhaugland
"""

#Script for removing black "curtains" on the training dataset PICCOLO.

from PIL import Image
import os

path_in = '/Users/mathiasrammhaugland/Documents/Masteroppgave/ThesisDatasets/set2_onlyPICCOLO/'
path_out = '/Users/mathiasrammhaugland/Documents/Masteroppgave/ThesisDatasets/set2_onlyPICCOLO/'


def main():
    sett = "test/"
    for mod in ["NBI_ref/"]:
        for nod in ["masks_old/", "polyps_old/"]:
            for file_path in os.listdir(path_in+mod+sett+nod):
                if nod=="masks_old/":
                    mode = "L"
                elif nod=="polyps_old/":
                    mode = "RGB"
                else: print("PROBLEM!")
                
                im = Image.open(path_in+mod+sett+nod+file_path).convert(mode)
                pol_nr = int(file_path[0]+file_path[1]+file_path[2])
                
                if pol_nr in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
                              32,33,34,
                              39,40,41,
                              57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76]:
                
                    left = 138
                    right = 740
                    bottom = 480
                    
                elif pol_nr in [23,24,25]:
                    
                    left = 454
                    right = 1804
                    bottom = 1080
                
                elif pol_nr in [26,27,28,29,30,31,
                                35,36,37,38,
                                42,43,44,45,46,47,48,49,50,51,52,53,54]:
                    left = 310
                    right = 1660
                    bottom = 1080
                
                else: print("PROBLEM2! "+pol_nr)
                print(im.size)
                imc = im.crop((left,0,right,bottom))
                print(imc.size)
                imc.save(path_out+mod+sett+nod.replace('_old','')+file_path)
                


if __name__ == '__main__':
    main()
