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
    def __init__(self, pattern_path, normals_map, output_path=None,
                 model_path=None):
        # Properties of the scene
        self.pattern_path = pattern_path
        self.normal_map_path = normals_map
        self.output_path = output_path
        self.model_path = model_path        
        self.def_position = 'fixed'
        self.objects = list()
        self.lights = list()
        self.cameras = list()
        self.materials = list()
        # Reset the default scene from blender
        bpy.ops.wm.read_factory_settings(use_empty=True)
        
        
    # Create the default scene
    def create_def_scene(self):
        # Define the constant parameters
        # TARGET
        TARGET_SIZE = (0.1, 0.1, 0.0015)
        # LIGHTS
        LIGHT_TYPE = "SPOT"
        LIGHT_POS = (-0.5, 0.3, 0.5)
        LIGHT_TARGET = np.array([0.0, 0.0, 0.0])
        LIGHT_INIT_ROT = np.array([0.0, 0.0, -1.0])
        LIGHT_ENERGY = 10.0
        LIGHT_SPOTSIZE = math.radians(25)
        LIGHT_SPOT_BLEND = 1.0
        LIGHT_SHAD_SPOT = 0.1
        CAM0_POS = (0.0, 0.0, 1.0)
        CAM1_POS = (0.259, 0.0, 0.966)
        CAM_TARGET = np.array([0.0, 0.0, 0.0])
        CAM_FOC_LENGTH = 50.0
        CAM_FSTOP = 8
        CAM_INIT_ROT = np.array([0.0, 0.0, -1.0])
        # CAMERAS
        # Add the default target
        target = self.add_cube(TARGET_SIZE)
        # Add the light panel
        # Calculate desired orientation of the light
        light_target_orient = LIGHT_TARGET - np.array(LIGHT_POS)
        # Calculate the rotation angle of the light
        light_angle = self.calc_rot_angle(LIGHT_INIT_ROT, light_target_orient)
        self.add_light(LIGHT_TYPE, pos=LIGHT_POS, orient=light_angle,
                       energy=LIGHT_ENERGY, spot_size=LIGHT_SPOTSIZE,
                       spot_blend=LIGHT_SPOT_BLEND, 
                       shadow_spot_size = LIGHT_SHAD_SPOT)
        # Add straight camera
        # Calculate desired orientation of the cam
        cam0_target_orient = CAM_TARGET - np.array(CAM0_POS) 
        cam0_target_dist = np.linalg.norm(cam0_target_orient)+1e-16
        cam_angle = self.calc_rot_angle(CAM_INIT_ROT, cam0_target_orient)
        cam0 = self.add_camera(pos=CAM0_POS, orient=cam_angle, fstop=CAM_FSTOP,
                               focal_length=CAM_FOC_LENGTH,
                               obj_distance=cam0_target_dist)  
        # Add cross camera
        cam1_target_orient = CAM_TARGET - np.array(CAM1_POS)
        cam1_target_dist = np.linalg.norm(cam1_target_orient)+1e-16
        cam_angle = self.calc_rot_angle(CAM_INIT_ROT, cam1_target_orient)
        cam1 = self.add_camera(pos=CAM1_POS, orient=cam_angle, fstop=CAM_FSTOP,
                               focal_length=CAM_FOC_LENGTH,
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
                     glossy_roughness=0.2, specular_strength=0.5, 
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
        bsdf_glossy.distribution = "MULTI_GGX"
        bsdf_glossy.inputs[1].default_value = glossy_roughness
        # Read an image to serve as a base texture for the specular reflection
        texImage = tree.nodes.new('ShaderNodeTexImage')
        texImage.image = bpy.data.images.load(self.pattern_path)
        bpy.data.images[0].colorspace_settings.name = 'Non-Color'
        tree.links.new(bsdf_glossy.inputs['Color'], 
                                texImage.outputs['Color'])
        # Read an image to serve as a normals map for the specular reflection
        normImage = tree.nodes.new('ShaderNodeTexImage')
        normImage.image = bpy.data.images.load(self.normal_map_path)
        bpy.data.images[0].colorspace_settings.name = 'Non-Color'
        normMap = tree.nodes.new('ShaderNodeNormalMap')
        normMap.inputs[0].default_value = specular_strength
        tree.links.new(normMap.inputs['Color'], 
                                normImage.outputs['Color'])
        tree.links.new(bsdf_glossy.inputs['Normal'], 
                                normMap.outputs['Normal'])
        # Add diffusive node (Diffuse BSDF)
        bsdf_diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
        # Set roughness
        bsdf_diffuse.inputs[1].default_value = diffuse_roughness
        # Link the pattern image to the colour of the diffusive reflection
        tree.links.new(bsdf_diffuse.inputs['Color'], 
                                texImage.outputs['Color'])
        # Add Mix shader node
        mix_shader = tree.nodes.new("ShaderNodeMixShader")
        mix_shader.inputs[0].default_value = shader_mix_ratio
        tree.links.new(mix_shader.inputs[1], 
                                bsdf_glossy.outputs['BSDF'])
        tree.links.new(mix_shader.inputs[2], 
                                bsdf_diffuse.outputs['BSDF'])
        # Link the Mix shader node with the Material output
        mat_output = tree.nodes["Material Output"]
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
      
pattern_path = r"E:\speckle\test\im.tiff"
normals_path = r"E:\speckle\test\grad.tiff"
model_path = r"E:\speckle\test\model_dev.blend"
output_path = r"E:\speckle\test\render_0.tiff"
output_path2 = r"E:\speckle\test\render_0.tiff"

a = VirtExp(pattern_path, normals_path, output_path, model_path)
a.create_def_scene()  
a.render_scene()
#a.set_renderer(a.cameras[1])
#a.render_scene(output_path2)    