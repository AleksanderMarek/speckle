"""
This script generates the default scene and renders two images.
The scene consists of a plate target and two cameras, the first one
is positioned perpendicularly to the target while the other is positioned
at an angle of 15 degrees to the target
"""

from speckle import VirtExp
import speckle

pattern_path = r"D:\Experiment Quality\blender_model\im.tiff"
normals_path = r"D:\Experiment Quality\blender_model\grad.tiff"
model_path = r"D:\Experiment Quality\blender_model\model_dev.blend"
output_path = r"D:\Experiment Quality\blender_model\render_0.tiff"
output_path2 = r"D:\Experiment Quality\blender_model\render_1.tiff"


# Generate speckle pattern
image_size = (1000, 1000)
target_size = (100, 100)
pixel_size_physical = 0.00345 # mm
imaging_dist = 1000
focal_length = 50
M = focal_length / (imaging_dist - focal_length)
pixel_size = pixel_size_physical/M
speckle_size = 4.0 * pixel_size
output_DPI = 600
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
# Generate blender scene
a = VirtExp(pattern_path, normals_path, output_path, model_path,
            objects_position="fixed")
# Set up the default scene
a.create_def_scene()  
# Render the scene with the perpendicular camera
a.render_scene()
# Switch the camera to the cross one and render the scene
a.set_renderer(a.cameras[1])
a.render_scene(output_path2) 

