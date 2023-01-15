import bpy
import math
import random
import numpy as np
import mathutils
import os

""" This script drive focus_tutorial.blend model and generates
rendered image of a speckle pattern

To do:
    -Finish commenting and clean the code
"""

def blender_render_model(output_path, pattern_path): 
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Set the original cube
    bpy.ops.mesh.primitive_cube_add(size = 0.1, enter_editmode = False,
                                    align = 'WORLD', location = (0, 0.05, 0))
    cube = bpy.data.objects["Cube"]
    # Spotlight properties
    # Shape and energy
    spot_variation = 5
    spot_size = math.radians(10 + random.uniform(-1, 1)*spot_variation)
    spot_blend = random.uniform(0, 1)
    spot_energy_variation = 20.0
    spot_energy = random.uniform(0, spot_energy_variation)
    shadow_spot_size = random.uniform(0.001, 0.05)
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
    light = bpy.data.lights.new(name="Spot", type='SPOT')
    # Create a new object to represent the light
    light_ob = bpy.data.objects.new(name="Spot", object_data=light)
    # Link the light object to the scene
    bpy.context.collection.objects.link(light_ob)
    # Position the light in the scene
    light_ob.location = spot_loc
    light_ob.rotation_euler = ang_euler
    light.spot_size = spot_size
    light.spot_blend = spot_blend
    light.energy = spot_energy
    light.shadow_soft_size = shadow_spot_size


    # Background light properties
    # Set light location
    light1 = bpy.data.lights.new(name="Light", type='SUN')
    # Create a new object to represent the light
    light1_ob = bpy.data.objects.new(name="Light", object_data=light1)
    # Link the light object to the scene
    bpy.context.collection.objects.link(light1_ob)
    light1_ob.location = (-1, -1, 0)
    light1_ob.rotation_euler = (math.radians(89.5), 0, math.radians(-33.4))
    light_variation = 5.0
    light_energy = 4.0 + random.uniform(-1, 1)*light_variation
    light1.energy = light_energy
    light1.angle = math.radians(12)

    # Camera properties
    # Add camera location
    fstop_variation = 30.0
    fstop = random.uniform(5.0, fstop_variation)
    focal_length = 50.0
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    bpy.context.collection.objects.link(camera)
    cam1 = bpy.data.cameras['Camera']
    camera.location = (0, -1, 0)
    camera.rotation_euler = (math.pi/2, 0, 0)
    cam1.dof.use_dof = True
    cam1.dof.aperture_fstop = fstop
    cam1.lens = focal_length
    cam1.sensor_width = 8.80
    cam1.sensor_height = 6.60
    cam1.dof.focus_object = bpy.data.objects["Cube"]

    

    # Add materials to the cube
    #im1 = bpy.ops.image.load("//..\\..\\..\\speckle\\images\\im4.tiff")
    #mat = bpy.data.materials.new(name="New_Mat")
    #mat.use_nodes = True
    #bsdf = mat.node_tree.nodes["Principled BSDF"]
    #texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    #texImage.image = bpy.data.images.load("D:\\Cool Projects\\Paperspace\\3-D Models\\Background.jpg")
    #mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    #ob = context.view_layer.objects.active
    mat = bpy.data.materials.new(name="Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs[6].default_value = 0.8 # metallic
    bsdf.inputs[7].default_value = 0.6 # specular
    bsdf.inputs[9].default_value = 0.23 # roughness

    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texImage = mat.node_tree.nodes["Image Texture"]
    texImage.image = bpy.data.images.load(pattern_path)
    mat.node_tree.links.new(bsdf.inputs['Base Color'], 
                            texImage.outputs['Color'])
    ob = bpy.context.view_layer.objects.active
    # Assign the material to the cube
    cube.data.materials.append(mat)
    bpy.data.objects['Cube'].select_set(True)
    #bpy.ops.outliner.item_activate(deselect_all=True)
    bpy.ops.object.editmode_toggle()
    bpy.ops.uv.cube_project(cube_size=20)



    # Render image and save
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.resolution_x = 2448
    bpy.context.scene.render.resolution_y = 2048
    bpy.context.scene.render.image_settings.file_format = 'TIFF'
    bpy.context.scene.render.image_settings.color_mode = 'BW'
    bpy.context.scene.render.image_settings.color_depth = '8'
    bpy.context.scene.render.image_settings.compression = 0
    bpy.context.scene.render.engine = 'CYCLES' #Working
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 300
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoising_input_passes = 'RGB_ALBEDO'
    bpy.context.scene.render.use_border = True
    bpy.ops.render.render(write_still=True)    
    #bpy.ops.wm.save_as_mainfile(
    #    filepath=r"E:\GitHub\speckle\development\blender\test1.blend")


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