"""
This script generates the default scene and renders two images.
The scene consists of a plate target and two cameras, the first one
is positioned perpendicularly to the target while the other is positioned
at an angle of 15 degrees to the target
"""

from speckle import VirtExp
import speckle
import numpy as np

pattern_path = r"D:\Experiment Quality\blender_model\im.tiff"
normals_path = r"D:\Experiment Quality\blender_model\grad.tiff"
model_path = r"D:\Experiment Quality\blender_model\model_dev.blend"
output_path = r"D:\Experiment Quality\blender_model\render_0_0.tiff"
output_path2 = r"D:\Experiment Quality\blender_model\render_0_1.tiff"
calib_path = r"D:\Experiment Quality\blender_model\calibration.caldat"
mesh_path = r"D:\GitHub\speckle\test_specimen\fullSpec.mesh"
displ_filepath = [r"D:\GitHub\speckle\test_specimen\fullSpec1.csv",
                  r"D:\GitHub\speckle\test_specimen\fullSpec2.csv",
                  r"D:\GitHub\speckle\test_specimen\fullSpec3.csv",
                  r"D:\GitHub\speckle\test_specimen\fullSpec4.csv",
                  r"D:\GitHub\speckle\test_specimen\fullSpec5.csv"]
# displ_filepath = []


# Generate speckle pattern
image_size = (2000, 3500)
target_size = (70, 160)
pixel_size_physical = 0.00345  # mm
imaging_dist = 1000
focal_length = 50
M = focal_length / (imaging_dist - focal_length)
pixel_size = pixel_size_physical/M
speckle_size = 4.0 * pixel_size
output_DPI = 600
# Check if the size of the speckle pattern is sufficient
if (output_DPI/25.4*target_size[0] > image_size[0]) or \
        (output_DPI/25.4*target_size[1] > image_size[1]):
    print("Warning: the resolution of the speckle pattern is too small!")
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
# Set up the scene
# Get default properties
p = a.get_default_params()
# CAMERAS
# Add the default target
target = a.add_FEA_part(mesh_path)
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
cam0_target_orient = p["cam0_target"] - np.array(p["cam0_pos"])
cam0_target_dist = np.linalg.norm(cam0_target_orient)+1e-16
cam_angle = a.calc_rot_angle(p["cam_init_rot"], cam0_target_orient)
cam0 = a.add_camera(pos=p["cam0_pos"], orient=cam_angle,
                    fstop=p["cam_fstop"],
                    focal_length=p["cam_foc_length"],
                    obj_distance=cam0_target_dist,
                    k1=0.0, p1=0.00)
# Add cross camera
cam1_target_orient = p["cam1_target"] - np.array(p["cam1_pos"])
cam1_target_dist = np.linalg.norm(cam1_target_orient)+1e-16
cam_angle = a.calc_rot_angle(p["cam_init_rot"], cam1_target_orient)
cam1 = a.add_camera(pos=p["cam1_pos"], orient=cam_angle,
                    fstop=p["cam_fstop"],
                    focal_length=p["cam_foc_length"],
                    obj_distance=cam1_target_dist)
a.rotate_around_z(cam1, 5.0)
# Define the material and assign it to the cube
a.add_material(target)
# Write the calibration file
a.generate_calib_file(cam0, cam1, calib_path)
# Set the renderer up and render image
a.set_renderer(cam0, n_samples=50)
# Add distortion to the model
a.add_image_distortion(cam0)
# Save the model
a.save_model()
# Render the scene with the perpendicular camera
a.render_scene(output_path)
# Switch the camera to the cross one and render the scene
a.set_renderer(cam1, n_samples=50)
# Add distortion to the model
a.add_image_distortion(cam1)
a.render_scene(output_path2)

# Deform images
for i, displ_file in enumerate(displ_filepath):
    # Update position of FE nodes
    a.deform_FEA_part(target, displ_file)
    # Add another frame to the animation
    a.set_new_frame(target)
    # Render the scene with the perpendicular camera
    def_path = f"D:\\Experiment Quality\\blender_model\\render_{i+1}_0.tiff"
    a.set_renderer(cam0, n_samples=50)
    a.add_image_distortion(cam0)
    a.render_scene(def_path)
    # Switch the camera to the cross one and render the scene
    def_path = f"D:\\Experiment Quality\\blender_model\\render_{i+1}_1.tiff"
    a.set_renderer(cam1, n_samples=50)
    a.add_image_distortion(cam1)
    a.render_scene(def_path)
# Save final model
a.save_model()
