#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:19:32 2022

@author: mathiasrammhaugland
"""

#Take in a .csv file width bbox heights and widths of gt, and plot height-width histogram
import csv
import matplotlib.pyplot as plt
import numpy as np

path = '/Users/mathiasrammhaugland/Documents/Masteroppgave/ThesisDatasets/set2_onlyPICCOLO/WLI/classification_files/'

csv_files = ['train_piccolo_wli.csv']

hw_ratios = []
wh_ratios = []

#put all ratios in list(s)
for file in csv_files:
    with open(path+file, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            if row[6] != 'xmax': #jump over first line
                box_width = int(row[6])-int(row[4])
                box_height = int(row[7])-int(row[5])
                hw_ratio = float(box_height/box_width)
                wh_ratio = float(box_width/box_height)
                
                hw_ratios.append(hw_ratio)
                wh_ratios.append(wh_ratio)

#plot histograms

plt.hist(hw_ratios, bins = 100, alpha=0.6, label='Height/width')
plt.hist(wh_ratios, bins = 100, alpha=0.6, label='Width/height')

plt.xticks(np.arange(0,5,step=0.4))
plt.title("Ratio histograms on PICCOLO training set")
plt.xlabel("Ratio")
plt.ylabel("Number of boxes")
plt.legend()
plt.show()

            
            