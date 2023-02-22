import speckle
import csv 
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 22:56:34 2023

@author: am3d18
"""

render_path = r"D:\Experiment Quality\rendered_images\render_3.tiff"
output_folder = r"D:\Experiment Quality\input_images"
summary_path = r"D:\Experiment Quality\summary.csv"

output_paths = speckle.trim_render_im(render_path, output_folder)

output_paths1 = r"D:\Experiment Quality\input_images\im_5.tiff"

imDef_inp = r"D:\Experiment Quality\ImDef\ImDef.mtind"
corr_inp = r"D:\Experiment Quality\ImDef\Job.m2inp"
noise_floor, mean_U = \
    speckle.image_deformation(output_paths1, imDef_inp, corr_inp)
record = ["pattern_1.tiff", "render_1.tiff", "im_5.tiff", 
         noise_floor, mean_U]    
with open(summary_path,'a') as fd:
    writer = csv.writer(fd)
    writer.writerow(record)

