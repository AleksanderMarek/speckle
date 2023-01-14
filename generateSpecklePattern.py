import speckle

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

image_size = (1000, 1000)
speckle_size = 4;
fft_threshold = 4
output_DPI = 600
target_size = (100, 100)
target_path = 'E:\\GitHub\\speckle\\speckle_images\\im4.tiff'

imaging_dist = 1000
focal_length = 50
M = focal_length / (imaging_dist - focal_length)
pixel_size_physical = 0.00345 # mm
pixel_size = pixel_size_physical/M
speckle_size = 5*pixel_size


pat1 = speckle.SpeckleImage(image_size, speckle_size)
pat1.set_physical_dim(target_size, speckle_size, output_DPI)
im1 = pat1.gen_pattern()
pat1.im_show()
pat1.im_save(target_path)