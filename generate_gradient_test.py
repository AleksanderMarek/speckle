import speckle
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 21:32:18 2023

@author: am3d18
"""

image_size = (1000, 1000)
speckle_size = 4;
fft_threshold = 4
output_DPI = 600
normal_filter = 5
output_path = r"D:\Experiment Quality\test\im.tiff"
grad_path = r"D:\Experiment Quality\test\grad.tiff"
render_path = r"D:\Experiment Quality\test\render1.tiff"
pat1 = speckle.SpeckleImage(image_size, speckle_size)
pat2 = speckle.SpeckleImage(image_size, 60, normal_map_filter=normal_filter)
im1 = pat1.gen_pattern()
pat2.gen_pattern()
pat2.pattern_gradient()
pat2.generate_norm_map()
pat1.im_save(output_path)
pat2.grad_save(grad_path)

speckle.blender_render_model(render_path, output_path, grad_path)