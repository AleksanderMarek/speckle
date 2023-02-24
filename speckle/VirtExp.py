"""
This class is used to produce and render a blender scene that represents an
object covered with a speckle pattern. It controls position of objects
such as lights and cameras, as well as texturing of the target and setting
of light scattering properties to create more realistic images

Writen by Aleksander Marek, aleksander.marek.pl@gmail.com
19/0/2023

"""

import bpy
import math
import random
import numpy as np
import mathutils

# Define class VirtExp that contains all the elements of the scene and renders
# the final image

class VirtExp:
    # Constructor
    def __init__(self, pattern_path, normal_map_path, output_path=None,
                 model_path=None, objects_position='fixed'):
        # Properties of the scene
        self.pattern_path = pattern_path
        self.normal_map_path = normal_map_path
        self.output_path = output_path
        self.model_path = model_path        
        self.objects_position = objects_position
        self.objects = list()
        self.lights = list()
        self.cameras = list()
        self.materials = list()
        # Reset the default scene from blender
        bpy.ops.wm.read_factory_settings(use_empty=True)
        
        
    # Create the default scene
    def create_def_scene(self):
        # Get default properties
        p = self.get_default_params()
        # CAMERAS
        # Add the default target
        target = self.add_cube(p["target_size"])
        # Add the light panel
        # Calculate desired orientation of the light
        light_target_orient = p["light_target"] - np.array(p["light_pos"])
        # Calculate the rotation angle of the light
        light_angle = self.calc_rot_angle(p["light_init_rot"], 
                                          light_target_orient)
        self.add_light(p["light_type"], pos=p["light_pos"], orient=light_angle,
                       energy=p["light_energy"], spot_size=p["light_spotsize"],
                       spot_blend=p["light_spot_blend"], 
                       shadow_spot_size = p["light_shad_spot"])
        # Add straight camera
        # Calculate desired orientation of the cam
        cam0_target_orient = p["cam0_target"] - np.array(p["cam0_pos"]) 
        cam0_target_dist = np.linalg.norm(cam0_target_orient)+1e-16
        cam_angle = self.calc_rot_angle(p["cam_init_rot"], cam0_target_orient)
        cam0 = self.add_camera(pos=p["cam0_pos"], orient=cam_angle, 
                               fstop=p["cam_fstop"], 
                               focal_length=p["cam_foc_length"],
                               obj_distance=cam0_target_dist)  
        # Add cross camera
        cam1_target_orient = p["cam1_target"] - np.array(p["cam1_pos"])
        cam1_target_dist = np.linalg.norm(cam1_target_orient)+1e-16
        cam_angle = self.calc_rot_angle(p["cam_init_rot"], cam1_target_orient)
        cam1 = self.add_camera(pos=p["cam1_pos"], orient=cam_angle, 
                               fstop=p["cam_fstop"],
                               focal_length=p["cam_foc_length"],
                               obj_distance=cam1_target_dist)         
        # Define the material and assign it to the cube
        self.add_material(target)
        # Set the renderer up and render image
        self.set_renderer(cam0)
        # Save the model
        self.save_model()
        
        
    # Method to place a new cube target in the scene
    def add_cube(self, cube_size):
        max_size = max(cube_size)
        # Calculate scaling to the desired shape
        scale = tuple([ii/max_size for ii in cube_size])
        bpy.ops.mesh.primitive_cube_add(size=max_size, 
                                        enter_editmode=False,
                                        align='WORLD',
                                        scale=scale,
                                        location=(0, 0, -cube_size[2]/2))
        cube = bpy.data.objects["Cube"]
        self.objects.append(cube)
        return cube
        
    # Method to place a new light in the scene
    def add_light(self, light_type, pos=(0, 0, 0), orient=(0, 0, 0), energy=0,
                  spot_size=0, spot_blend=0,shadow_spot_size=0):
        if light_type == "SPOT":
            # Assign properties to the spotlight
            light = bpy.data.lights.new(name="Spot", type='SPOT')
            # Create a new object to represent the light
            light_ob = bpy.data.objects.new(name="Spot", object_data=light)
            # Link the light object to the scene
            bpy.context.collection.objects.link(light_ob)
            # Position the light in the scene
            light_ob.location = pos
            light_ob.rotation_euler = orient
            # DEfine the properties of the lamp
            light.energy = energy
            light.spot_size = spot_size
            light.spot_blend = spot_blend
            light.shadow_soft_size = shadow_spot_size
            # Add the object to the scene collection
            self.lights.append(light_ob)
            return light_ob
        
    # Method to add a camera to the scene
    def add_camera(self, pos=(0, 0, 0), orient=(0, 0, 0), obj_distance=None, 
                   fstop=0, focal_length=50.0, sensor_size=(8.8, 6.6),
                   sensor_px=(2448, 2048)):    
        # Create new data and object for the camera
        cam1 = bpy.data.cameras.new("Camera")
        camera = bpy.data.objects.new("Camera", cam1)
        # Introduce the camera to the scene
        bpy.context.collection.objects.link(camera)
        # Define camera properties
        camera.location = pos
        camera.rotation_euler = orient
        cam1.lens = focal_length
        cam1.sensor_width = sensor_size[0]
        cam1.sensor_height = sensor_size[1]
        # Add depth of field simulations
        if obj_distance is not None:
            cam1.dof.focus_distance = obj_distance  
            cam1.dof.use_dof = True
            cam1.dof.aperture_fstop = fstop
        # Add custom field and store number of pixels    
        camera["sensor_px"] = sensor_px
        # Append to the camera list 
        self.cameras.append(camera)
        return camera
    
    # Method to define a new material and add it to the selected object
    def add_material(self, target, 
                     glossy_roughness=0.2, specular_strength=1.0, 
                     diffuse_roughness=0.2, shader_mix_ratio=0.7):
        # Make a new material
        mat = bpy.data.materials.new(name="Material")
        mat.use_nodes = True
        tree =  mat.node_tree
        # Remove the default BSDF node
        bsdf = tree.nodes["Principled BSDF"]
        tree.nodes.remove(bsdf)
        # Define specular node (Glossy BSDF)
        bsdf_glossy = mat.node_tree.nodes.new("ShaderNodeBsdfGlossy")
        bsdf_glossy.location = (-250, 408)
        bsdf_glossy.distribution = "MULTI_GGX"
        bsdf_glossy.inputs[1].default_value = glossy_roughness
        # Read an image to serve as a base texture for the specular reflection
        texImage = tree.nodes.new('ShaderNodeTexImage')
        texImage.location = (-825, 63)
        texImage.image = bpy.data.images.load(self.pattern_path)
        bpy.data.images[0].colorspace_settings.name = 'Non-Color'
        tree.links.new(bsdf_glossy.inputs['Color'], 
                                texImage.outputs['Color'])
        # Read an image to serve as a normals map for the specular reflection
        norm_image = tree.nodes.new('ShaderNodeTexImage')
        norm_image.location = (-825, 373)
        norm_image.image = bpy.data.images.load(self.normal_map_path)
        bpy.data.images[0].colorspace_settings.name = 'Non-Color'
        norm_map = tree.nodes.new('ShaderNodeNormalMap')
        norm_map.location = (-525, 425)
        norm_map.inputs[0].default_value = specular_strength
        norm_map.uv_map = "UVMap"
        tree.links.new(norm_map.inputs['Color'], 
                                norm_image.outputs['Color'])
        tree.links.new(bsdf_glossy.inputs['Normal'], 
                                norm_map.outputs['Normal'])
        # Add diffusive node (Diffuse BSDF)
        bsdf_diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
        bsdf_diffuse.location = (-424, 21)
        # Set roughness
        bsdf_diffuse.inputs[1].default_value = diffuse_roughness
        # Link the pattern image to the colour of the diffusive reflection
        tree.links.new(bsdf_diffuse.inputs['Color'], 
                                texImage.outputs['Color'])
        # Add Mix shader node
        mix_shader = tree.nodes.new("ShaderNodeMixShader")
        mix_shader.location = (42, 340)
        mix_shader.inputs[0].default_value = shader_mix_ratio
        tree.links.new(mix_shader.inputs[1], 
                                bsdf_glossy.outputs['BSDF'])
        tree.links.new(mix_shader.inputs[2], 
                                bsdf_diffuse.outputs['BSDF'])
        # Link the Mix shader node with the Material output
        mat_output = tree.nodes["Material Output"]
        mat_output.location = (250, 340)
        tree.links.new(mat_output.inputs['Surface'], 
                                mix_shader.outputs['Shader'])
        # Separate cube into faces for texture mapping
        ob = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active
        # Assign the material to the cube
        target.data.materials.append(mat)
        target.select_set(True)
        #bpy.ops.outliner.item_activate(deselect_all=True)
        bpy.ops.object.editmode_toggle()
        bpy.ops.uv.cube_project(scale_to_bounds=True)
        bpy.ops.object.editmode_toggle()
        # Add the material to the list
        self.materials.append(mat)
        return mat
        
    # Define method to set all the properties of the renderer    
    def set_renderer(self, cam):
        scene = bpy.context.scene
        scene.camera = cam
        scene.render.filepath = self.output_path
        scene.render.resolution_x = cam["sensor_px"][0]
        scene.render.resolution_y = cam["sensor_px"][1]
        scene.render.image_settings.file_format = 'TIFF'
        scene.render.image_settings.color_mode = 'BW'
        scene.render.image_settings.color_depth = '8'
        scene.render.image_settings.compression = 0
        scene.render.engine = 'CYCLES' #Working
        scene.cycles.device = 'GPU'
        scene.cycles.samples = 100
        scene.cycles.use_denoising = True
        scene.cycles.denoising_input_passes = 'RGB_ALBEDO'
        scene.render.use_border = True
        
    # Method to calculate rotation between two 3D vectors using Euler angles
    def calc_rot_angle(self, dir1, dir2):
        # Normalise the directions
        dir1 /= np.linalg.norm(dir1) + 1e-16
        dir2 /= np.linalg.norm(dir2) + 1e-16
        # Calculate the rotation matrix using Rodriguez' formula
        v = np.cross(dir1, dir2)
        s = np.linalg.norm(v)+1e-16
        c = np.dot(dir1, dir2)
        v_skew = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
            ])
        rot_mat = np.eye(3) + v_skew + np.dot(v_skew, v_skew)*(1-c)/s**2
        rot_mat = mathutils.Matrix(rot_mat)
        ang_euler = rot_mat.to_euler("XYZ")
        return ang_euler
    
    # Method to remove an object from the scene
    def remove_object(self, obj):
        bpy.data.objects.remove(obj, do_unlink=True,
                                do_id_user=True, do_ui_user=True)

    # Method to render the scene
    def render_scene(self, filepath=None):
        if filepath is not None:
            bpy.context.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
    
    # Save the blender model a file
    def save_model(self):
        if self.model_path is not None:
            bpy.ops.wm.save_as_mainfile(filepath=self.model_path)
            
    # Get properties for the default scene
    def get_default_params(self):
        # Define the dictionary
        props = {}
        # Define constant parameters
        props["target_size"] = (0.1, 0.1, 0.0015)
        props["light_type"] = "SPOT"
        props["light_init_rot"] = np.array([0.0, 0.0, -1.0])
        props["cam_foc_length"] = 50.0
        props["cam_init_rot"] = np.array([0.0, 0.0, -1.0])
        props["cam0_pos"] = (0.0, 0.0, 1.0)
        props["cam0_target"] = np.array([0.0, 0.0, 0.0])
        # Define parameters that are allowed to change
        # Constant
        if self.objects_position == 'fixed':
            props["light_pos"] = (-0.5, 0.3, 0.5)
            props["light_target"] = np.array([0.0, 0.0, 0.0])
            props["light_energy"] = 20.0
            props["light_spotsize"] = math.radians(25)
            props["light_spot_blend"] = 0.5
            props["light_shad_spot"] = 0.01
            props["cam1_pos"] = (0.259, 0.0, 0.966)
            props["cam1_target"] = np.array([0.0, 0.0, 0.0])
            props["cam_fstop"] = 8.0
            props["cam1_target"] = np.array([0.0, 0.0, 0.0])
        # Random    
        elif self.objects_position == 'random':
            # Control parameters
            polar_ang_variation = 60
            azim_ang_variation = 30 
            spot_dist_mean = 0.5
            spot_dist_variation = 0.2
            target_range = 0.075
            spot_energy_mean = 20.0
            spot_energy_variation = 10.0
            spot_ang_mean = 35.0
            spot_ang_variation = 10.0
            spot_size_min = 0.0
            spot_size_max = 0.01
            cam1_dist_mean = 1.0
            cam1_dist_variation = 0.005
            cam1_ang_mean = 15.0
            cam1_ang_variation = 5.0
            cam1_target_variation = 0.03
            fstop_min = 4.0
            fstop_max = 11.0
            # Position of the spotlight
            polar_ang = math.radians(
                random.uniform(-1, 1) * polar_ang_variation)
            azim_ang = math.radians(
                random.uniform(-1, 1) * azim_ang_variation)
            spot_dist = spot_dist_mean \
                + random.uniform(-1, 1) * spot_dist_variation
            x = spot_dist * math.cos(polar_ang) * math.sin(azim_ang)
            y = spot_dist * math.sin(polar_ang) * math.sin(azim_ang)
            z = spot_dist * math.cos(azim_ang)
            props["light_pos"] = (x, y, z)
            # Light target
            props["light_target"] = (
                random.uniform(-target_range, target_range),
                0,
                random.uniform(-target_range, target_range)
                )
            # Light energy
            props["light_energy"] = random.normalvariate(spot_energy_mean,
                                                      spot_energy_variation)
            # Spot size
            props["light_spotsize"] = math.radians(
                spot_ang_mean + random.uniform(-1, 1) * spot_ang_variation)
            # Spot blend
            props["light_spot_blend"] = random.uniform(0.0, 1.0)
            # Light shadow spot
            props["light_shad_spot"] = random.uniform(spot_size_min,
                                                      spot_size_max)
            # Position of cross camera
            cam1_dist = random.normalvariate(cam1_dist_mean, 
                                          cam1_dist_variation)
            cam1_ang = math.radians(
                random.normalvariate(cam1_ang_mean, cam1_ang_variation)
                )
            props["cam1_pos"] = (
                cam1_dist*math.sin(cam1_ang),
                0.0,
                cam1_dist*math.cos(cam1_ang))
            # Cross camera target
            props["cam1_target"] = (
                random.normalvariate(0.0, cam1_target_variation),
                random.normalvariate(0.0, cam1_target_variation),
                0.0)
            # Aperture
            props["cam_fstop"] = random.uniform(fstop_min, fstop_max)
        return props
               