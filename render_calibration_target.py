"""
This script generates the default scene and renders two images.
The scene consists of a plate target and two cameras, the first one
is positioned perpendicularly to the target while the other is positioned
at an angle of 15 degrees to the target
"""

from speckle import VirtExp
import speckle
import numpy as np

pattern_path = r"D:\Experiment Quality\blender_model\5mm.png"
normals_path = r"D:\Experiment Quality\blender_model\10mm.png"
model_path = r"D:\Experiment Quality\blender_model\model_calib.blend"
output_path = r"D:\Experiment Quality\Correlation test\calibration\render_0.tiff"
output_path2 = r"D:\Experiment Quality\Correlation test\calibration\render_1.tiff"


# Generate speckle pattern
image_size = (2500, 2500)
target_size = (100, 100)
calib_plate_size = (0.200, 0.150, 0.005)
pixel_size_physical = 0.00345 # mm
imaging_dist = 1000
focal_length = 50
M = focal_length / (imaging_dist - focal_length)
pixel_size = pixel_size_physical/M
speckle_size = 4.0 * pixel_size
output_DPI = 600
n_samples = 20
# Generate blender scene
# Generate blender scene
a = VirtExp(pattern_path, normals_path, output_path, model_path,
            objects_position="fixed")
# Set up the scene
# Get default properties
p = a.get_default_params()
# CAMERAS
# Add the default target
target = a.add_rect_target(calib_plate_size)
# Add the light panel
# Calculate desired orientation of the light
light_target_orient = p["light_target"] - np.array(p["light_pos"])
# Calculate the rotation angle of the light
light_angle = a.calc_rot_angle(p["light_init_rot"],
                               light_target_orient)
a.add_light(p["light_type"], pos=p["light_pos"], orient=light_angle,
            energy=p["light_energy"], spot_size=p["light_spotsize"],
            spot_blend=p["light_spot_blend"],
            shadow_spot_size=p["light_shad_spot"])
# Add straight camera
# Calculate desired orientation of the cam
cam0_pos = p["cam0_pos"]
# cam0_pos = np.array([-0.153, 0.002, 1.001])
cam0_target_orient = p["cam0_target"] - cam0_pos
cam0_target_dist = np.linalg.norm(cam0_target_orient)+1e-16
cam_angle = a.calc_rot_angle(p["cam_init_rot"], cam0_target_orient)
cam0 = a.add_camera(pos=cam0_pos, orient=cam_angle,
                    fstop=p["cam_fstop"],
                    focal_length=p["cam_foc_length"],
                    obj_distance=cam0_target_dist,
                    k1=0.0, p1=0.00)
# Add cross camera
cam1_pos = p["cam1_pos"]
# cam1_pos = np.array([0.174, -0.002, 1.015])
cam1_target_orient = p["cam1_target"] - cam1_pos
cam1_target_dist = np.linalg.norm(cam1_target_orient)+1e-16
cam_angle = a.calc_rot_angle(p["cam_init_rot"], cam1_target_orient)
cam1 = a.add_camera(pos=cam1_pos, orient=cam_angle,
                    fstop=p["cam_fstop"],
                    focal_length=p["cam_foc_length"],
                    obj_distance=cam1_target_dist)
# a.rotate_around_z(cam1, 0.5)
# a.rotate_around_z(cam0, 0.1)
# Define the material and assign it to the cube
a.add_material(target)
# Write the calibration file
# Set the renderer up and render image
a.set_renderer(cam0, n_samples=n_samples)
# Add distortion to the model
a.add_image_distortion(cam0)
# Save the model
a.save_model()
# Render the scene with the perpendicular camera
a.render_scene(output_path)
# Switch the camera to the cross one and render the scene
a.set_renderer(cam1, n_samples=n_samples)
# Add distortion to the model
a.add_image_distortion(cam1)
a.render_scene(output_path2)

