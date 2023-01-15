import speckle
import random
import csv 

""" This script generates training images for qe-net. First a synthetic
speckle pattern is generated using a FFT-based algorithm. Then, the pattern
is used in blender to simulate more realistic experimental conditions such as
lighting, blur etc. Finally, the image rendered in blender is split into 
500 x 500 px training images that will be used to calculate noise floor and 
train the net
"""

""" To do:
    -Develop robust loging of generation parameters so that each training image
    can be linked directly to the optimised speckle generator input and 
    blender input
"""    

# Define properties of the original speckle pattern
imaging_dist = 1000
focal_length = 50
M = focal_length / (imaging_dist - focal_length)
image_size = (1000, 1000)
#speckle_size = 4;
fft_threshold = 4
output_DPI = 600
target_size = (100, 100)
pixel_size_physical = 0.00345 # mm
pixel_size = pixel_size_physical/M
raw_speckle_folder = r'D:\Experiment Quality\speckle_images'
raw_speckle_prefix = 'pattern'
n_speckles = 25

# Define properties of the render
render_folder = r"D:\Experiment Quality\rendered_images"
render_prefix = 'render'
n_renders = 25

# Define properties of training images
output_folder = 'D:\Experiment Quality\input_images'

# Define paths to MatchID input files
imDef_inp = r"D:\Experiment Quality\ImDef\ImDef.mtind"
corr_inp = r"D:\Experiment Quality\ImDef\Job.m2inp"

# Define ledger of labels
summary_path = r"D:\Experiment Quality\summary.csv"


for ii in range(n_speckles):
    raw_speckle_path = speckle.generate_output_name(
        raw_speckle_folder, raw_speckle_prefix)
    speckle_size = random.uniform(3,7)*pixel_size
    pat1 = speckle.SpeckleImage(image_size, speckle_size)
    pat1.set_physical_dim(target_size, speckle_size, output_DPI)
    im1 = pat1.gen_pattern()
    pat1.im_save(raw_speckle_path)
    for jj in range(n_renders):
        render_path = speckle.generate_output_name(
            render_folder, render_prefix)
        speckle.blender_render_model(render_path, raw_speckle_path)
        output_paths = speckle.trim_render_im(render_path, output_folder)
        for kk in range(len(output_paths)):
            noise_floor, mean_U = \
                speckle.image_deformation(output_paths[kk], 
                                          imDef_inp, corr_inp)
            record = [raw_speckle_path, render_path, 
                      output_paths[kk], noise_floor, mean_U]    
            with open(summary_path, 'a', newline='') as fd:
                writer = csv.writer(fd)
                writer.writerow(record)
            





