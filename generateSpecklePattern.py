import speckle


image_size = (1000, 1000)
speckle_size = 4;
fft_threshold = 4
output_DPI = 450
target_path = 'E:\\GitHub\\speckle\\images\\im3.tiff'


pat1 = speckle.SpeckleImage(image_size, speckle_size)
im1 = pat1.gen_pattern()
pat1.im_show()
pat1.im_save(target_path)