import speckle


image_size = (1000, 1000)
speckle_size = 4;
fft_threshold = 4
output_DPI = 600
target_size = (100, 100)
target_path = 'E:\\GitHub\\speckle\\images\\im4.tiff'

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