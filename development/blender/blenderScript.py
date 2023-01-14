import bpy
import math
import random
import numpy as np
import mathutils
import os

""" This script drive focus_tutorial.blend model and generates
rendered image of a speckle pattern
"""

def blender_render_model(output_path, pattern_path):
    # Spotlight properties
    # Shape and energy
    spot_variation = 5
    spot_size = math.radians(10 + random.uniform(-1, 1)*spot_variation)
    spot_blend = random.uniform(0, 1)
    spot_energy_variation = 25.0
    spot_energy = 35.0 + random.uniform(-1, 1)*spot_energy_variation
    # Position
    polar_ang_variation = 60
    azim_ang_variation = 30
    R_variation = 0.2
    polar_ang = math.radians(-90 + random.uniform(-1, 1)*polar_ang_variation)
    azim_ang = math.radians(30 + random.uniform(-1, 1)*azim_ang_variation)
    R = 0.5 + random.uniform(-1, 1)*R_variation
    x = R*math.cos(polar_ang)*math.sin(azim_ang)
    y = R*math.sin(polar_ang)*math.sin(azim_ang)
    z = R*math.cos(azim_ang)
    spot_loc = (x, y, z)
    target = (random.uniform(-0.05, 0.05),
              0,
              random.uniform(-0.05, 0.05))
    # Rotation of the spotlight
    ray_direction = np.array(
        [target[0] - spot_loc[0],
        target[1] - spot_loc[1],
        target[2] - spot_loc[2]]
        )
    ray_direction /= (np.linalg.norm(ray_direction) + 1e-16)
    initial_dir = np.array([0, 0, -1])
    v = np.cross(ray_direction, initial_dir)
    s = np.linalg.norm(v)
    c = np.dot(ray_direction, initial_dir)
    v_skew = np.array(
        [[0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]]
        )    
    Rot = np.eye(3) + v_skew + np.dot(v_skew, v_skew)*(1-c)/s**2
    Rot_mat = mathutils.Matrix(Rot.T)
    ang_euler = Rot_mat.to_euler('XYZ')
    # Assign properties to the spotlight
    bpy.data.objects['Spot'].location = spot_loc
    bpy.data.objects['Spot'].rotation_euler = ang_euler
    spot1 = bpy.data.lights['Spot']
    spot1.spot_size = spot_size
    spot1.spot_blend = spot_blend
    spot1.energy = spot_energy

    # Background light properties
    # Set light location
    light_variation = 10.0
    light_energy = 5.0 + random.uniform(-1, 1)*light_variation
    light1 = bpy.data.lights['Light']
    light1.energy = light_energy

    # Camera properties
    # Add camera location
    fstop_variation = 16.0
    fstop = random.uniform(0, fstop_variation)
    focal_length = 50.0
    cam1 = bpy.data.cameras['Camera']
    cam1.dof.aperture_fstop = fstop
    cam1.lens = focal_length

    # Add materials to the cube
    #im1 = bpy.ops.image.load("//..\\..\\..\\speckle\\images\\im4.tiff")
    #mat = bpy.data.materials.new(name="New_Mat")
    #mat.use_nodes = True
    #bsdf = mat.node_tree.nodes["Principled BSDF"]
    #texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    #texImage.image = bpy.data.images.load("D:\\Cool Projects\\Paperspace\\3-D Models\\Background.jpg")
    #mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    #ob = context.view_layer.objects.active
    mat = bpy.data.materials["Material"]
    texImage = mat.node_tree.nodes["Image Texture"]
    texImage.image = bpy.data.images.load(pattern_path)


    # Render image and save
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = 'TIFF'
    bpy.context.scene.render.image_settings.color_mode = 'BW'
    bpy.context.scene.render.image_settings.color_depth = '8'
    bpy.context.scene.render.image_settings.compression = 0
    bpy.ops.render.render(write_still=True)    


"""    
for ii in range(1,5):
    k = ii
    output_path = f'E:\\GitHub\\speckle\\rendered_images\\test_{k}.tiff'
    pattern_path = f"E:\\GitHub\speckle\\speckle_images\\im{k}.tiff"
    while os.path.exists(output_path):
        k += 1
        output_path = f'E:\\GitHub\\speckle\\development\\blender\\test_{k}.tiff'
    blender_render_model(output_path, pattern_path)
    
"""        