"""
This script generates the default scene and renders two images.
The scene consists of a plate target and two cameras, the first one
is positioned perpendicularly to the target while the other is positioned
at an angle of 15 degrees to the target
"""

from speckle import VirtExp
import speckle
import os

# Define paths for output files
output_folder = os.path.join(os.getcwd(), 'default_scene')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
pattern_path = os.path.join(output_folder, "im.tiff")
normals_path = os.path.join(output_folder, "grad.tiff")
model_path = os.path.join(output_folder, "model.blend")
output_path = os.path.join(output_folder, "render_0_0.tiff")
output_path2 = os.path.join(output_folder, "render_0_1.tiff")
calib_path = os.path.join(output_folder, "calibration.caldat")


# PARAMETERS
image_size = (2500, 2500)  # Not relevant in this case
target_size = (100, 100)  # Size of the default target
pixel_size_physical = 0.00345  # mm / Manta G504-b
imaging_dist = 1000  # mm / distance from the camera to the target
focal_length = 50  # mm
# Calculate magnification
M = focal_length / (imaging_dist - focal_length)
# Get mm to pix ratio in the image
pixel_size = pixel_size_physical/M
# Generate speckle size whose physical size is around 4 pixels
speckle_size = 4.0 * pixel_size
# Speckle resolution
output_DPI = 600

# %% Generate speckle pattern
pat1 = speckle.SpeckleImage(image_size, speckle_size)
pat1.set_physical_dim(target_size, speckle_size, output_DPI)
im1 = pat1.gen_pattern()
pat1.pattern_gradient()
pat1.im_save(pattern_path)
# Generate normals map
normals_map = speckle.SpeckleImage(pat1.image_size, 60, normal_map_filter=1)
normals_map.gen_pattern()
normals_map.pattern_gradient()
normals_map.generate_norm_map(binary_map=pat1.gradient)
normals_map.grad_save(normals_path)

# %% Generate blender scene
a = VirtExp(pattern_path, normals_path, output_path, model_path,
            objects_position="fixed")
# Set up the default scene
a.create_def_scene()
cam0 = a.cameras[0]
cam1 = a.cameras[1]
# Render the scene with the perpendicular camera
a.render_scene()
a.add_image_distortion(cam0, output_path)
# Switch the camera to the cross one and render the scene
a.set_renderer(cam1)
a.render_scene(output_path2)
a.add_image_distortion(cam1, output_path2)
# Write the calibration file
a.generate_calib_file(cam0, cam1, calib_path)
# Save the model to *.blend file for inspection
a.save_model()
