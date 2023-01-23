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

def blender_render_model(output_path, pattern_path, normal_map_path): 
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Define constants
    DIFFUSE_ROUGHNESS = 0.5
    GLOSSY_ROUGHNESS = 0.2
    SHADER_MIX_RATIO_MAX = 0.95
    SHADER_MIX_RATIO_MIN = 0.4
    SPOT_DIST_MEAN = 0.5
    SPOT_DIST_VARIATION = 0.2
    SPOT_ANG_MEAN = 35
    SPOT_ANG_VARIATION = 10
    SPOT_ENERGY_MEAN = 20
    SPOT_ENERGY_VARIATION = 10
    SPOT_SIZE_MIN = 0.0
    SPOT_SIZE_MAX = 0.01
    FSTOP_MIN = 4.0
    FSTOP_MAX = 16.0
    SPEC_STR_MIN = 0.1
    SPEC_STR_MAX = 1.0
    # Set the original cube
    bpy.ops.mesh.primitive_cube_add(size = 0.1, enter_editmode = False,
                                    align = 'WORLD', location = (0, 0.05, 0))
    cube = bpy.data.objects["Cube"]
    # Spotlight properties
    # Shape and energy
    spot_size = math.radians(SPOT_ANG_MEAN \
                             + random.uniform(-1, 1)*SPOT_ANG_VARIATION)
    spot_blend = random.uniform(0, 1)
    #spot_energy = SPOT_ENERGY_MEAN \
    #    +random.uniform(-1, 1)*SPOT_ENERGY_VARIATION
    spot_energy = random.normalvariate(SPOT_ENERGY_MEAN, 
                                       SPOT_ENERGY_VARIATION)
    shadow_spot_size = random.uniform(SPOT_SIZE_MIN, SPOT_SIZE_MAX)
    # Position
    polar_ang_variation = 60
    azim_ang_variation = 30  
    polar_ang = math.radians(-90 + random.uniform(-1, 1) * polar_ang_variation)
    azim_ang = math.radians(90 + random.uniform(-1, 1) * azim_ang_variation)
    spot_dist = SPOT_DIST_MEAN + random.uniform(-1, 1)*SPOT_DIST_VARIATION
    x = spot_dist * math.cos(polar_ang) * math.sin(azim_ang)
    y = spot_dist * math.sin(polar_ang) * math.sin(azim_ang)
    z = spot_dist * math.cos(azim_ang)
    spot_loc = (x, y, z)
    target = (random.uniform(-0.075, 0.075),
              0,
              random.uniform(-0.075, 0.075))
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
    #bpy.context.collection.objects.link(light1_ob)
    light1_ob.location = (-1, -1, 0)
    light1_ob.rotation_euler = (math.radians(89.5), 0, math.radians(-33.4))
    light_variation = 5.0
    light_energy = 4.0 + random.uniform(-1, 1)*light_variation
    light1.energy = light_energy
    light1.angle = math.radians(12)

    # Camera properties
    # Add camera location
    fstop = random.uniform(FSTOP_MIN, FSTOP_MAX)
    focal_length = 50.0
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    bpy.context.collection.objects.link(camera)
    cam1 = bpy.data.cameras['Camera']
    camera.location = (0, -1, 0)
    camera.rotation_euler = (math.pi/2, 0, 0)
    cam1.dof.use_dof = True
    cam1.dof.aperture_fstop = fstop
    cam1.lens = focal_length
    cam1.sensor_width = 8.46
    cam1.sensor_height = 7.09
    # cam1.dof.focus_object = bpy.data.objects["Cube"]
    cam1.dof.focus_distance = 1.0
    
    # Add cross camera
    cross_angle = math.radians(20)
    cross_dist = 1.0
    camera2 = bpy.data.objects.new("Camera_off", 
                                  bpy.data.cameras.new("Camera_off"))
    bpy.context.collection.objects.link(camera2)
    cam2 = bpy.data.cameras['Camera_off']
    camera2.location = (cross_dist*math.sin(cross_angle), 
                        -cross_dist*math.cos(cross_angle), 
                        0.0)
    camera2.rotation_euler = (math.pi/2, 0, math.sin(cross_angle))
    cam2.dof.use_dof = True
    cam2.dof.aperture_fstop = fstop
    cam2.lens = focal_length
    cam2.sensor_width = 8.80
    cam2.sensor_height = 6.60
    cam2.dof.focus_object = bpy.data.objects["Cube"]
    
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
    bsdf_glossy.inputs[1].default_value = GLOSSY_ROUGHNESS
    # Read an image to serve as a base texture for the specular reflection
    texImage = tree.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(pattern_path)
    bpy.data.images[0].colorspace_settings.name = 'Non-Color'
    tree.links.new(bsdf_glossy.inputs['Color'], 
                            texImage.outputs['Color'])
    # Read an image to serve as a normals map for the specular reflection
    normImage = tree.nodes.new('ShaderNodeTexImage')
    normImage.image = bpy.data.images.load(normal_map_path)
    bpy.data.images[0].colorspace_settings.name = 'Non-Color'
    normMap = tree.nodes.new('ShaderNodeNormalMap')
    specularStrength = random.uniform(SPEC_STR_MIN, SPEC_STR_MAX)
    normMap.inputs[0].default_value = specularStrength
    tree.links.new(normMap.inputs['Color'], 
                            normImage.outputs['Color'])
    tree.links.new(bsdf_glossy.inputs['Normal'], 
                            normMap.outputs['Normal'])
    # Add diffusive node (Diffuse BSDF)
    bsdf_diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    # Set roughness
    bsdf_diffuse.inputs[1].default_value = DIFFUSE_ROUGHNESS
    # Link the pattern image to the colour of the diffusive reflection
    tree.links.new(bsdf_diffuse.inputs['Color'], 
                            texImage.outputs['Color'])
    # Add Mix shader node
    mix_shader = tree.nodes.new("ShaderNodeMixShader")
    shader_mix_ratio = random.uniform(SHADER_MIX_RATIO_MIN,
                                      SHADER_MIX_RATIO_MAX)
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
    cube.data.materials.append(mat)
    bpy.data.objects['Cube'].select_set(True)
    #bpy.ops.outliner.item_activate(deselect_all=True)
    bpy.ops.object.editmode_toggle()
    bpy.ops.uv.cube_project(scale_to_bounds=True)



    # Render image and save
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.filepath = output_path
    scene.render.resolution_x = 2452
    scene.render.resolution_y = 2056
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
    bpy.ops.render.render(write_still=True)    
    bpy.ops.wm.save_as_mainfile(
        filepath=r"D:\Experiment Quality\test\test1.blend")


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