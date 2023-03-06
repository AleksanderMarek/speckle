"""
This class is used to produce and render a blender scene that represents an
object covered with a speckle pattern. It controls position of objects
such as lights and cameras, as well as texturing of the target and setting
of light scattering properties to create more realistic images

Writen by Aleksander Marek, aleksander.marek.pl@gmail.com
19/01/2023

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
        bpy.context.scene.unit_settings.length_unit = 'MILLIMETERS'

    # Create the default scene

    def create_def_scene(self):
        # Get default properties
        p = self.get_default_params()
        # CAMERAS
        # Add the default target
        # target = self.add_cube(p["target_size"])
        target = self.add_rect_target(p["target_size"])
        # Add the light panel
        # Calculate desired orientation of the light
        light_target_orient = p["light_target"] - np.array(p["light_pos"])
        # Calculate the rotation angle of the light
        light_angle = self.calc_rot_angle(p["light_init_rot"],
                                          light_target_orient)
        self.add_light(p["light_type"], pos=p["light_pos"], orient=light_angle,
                       energy=p["light_energy"], spot_size=p["light_spotsize"],
                       spot_blend=p["light_spot_blend"],
                       shadow_spot_size=p["light_shad_spot"])
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
        self.add_camera(pos=p["cam1_pos"], orient=cam_angle,
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

    # Method to place a new rectangular target in the scene
    def add_rect_target(self, rect_size):
        nodes = [(-rect_size[0]/2, rect_size[1]/2, 0),
                 (-rect_size[0]/2, -rect_size[1]/2, 0),
                 (rect_size[0]/2, -rect_size[1]/2, 0),
                 (rect_size[0]/2, rect_size[1]/2, 0)]
        elements = [(0, 1, 2, 3)]
        thickness = rect_size[2]
        # Create mesh
        mesh = bpy.data.meshes.new("part")
        mesh.from_pydata(nodes, [], elements)
        obj = bpy.data.objects.new("specimen", mesh)
        bpy.context.scene.collection.objects.link(obj)
        part = bpy.data.objects["specimen"]
        # Add thickness to the mesh
        part.modifiers.new(name='solidify', type='SOLIDIFY')
        part.modifiers["solidify"].thickness = thickness
        # Return the object
        self.objects.append(part)
        return part

    # Method to place a new light in the scene
    def add_light(self, light_type, pos=(0, 0, 0), orient=(1, 0, 0, 0),
                  energy=0, spot_size=0, spot_blend=0, shadow_spot_size=0):
        if light_type == "SPOT":
            # Assign properties to the spotlight
            light = bpy.data.lights.new(name="Spot", type='SPOT')
            # Create a new object to represent the light
            light_ob = bpy.data.objects.new(name="Spot", object_data=light)
            # Link the light object to the scene
            bpy.context.collection.objects.link(light_ob)
            # Position the light in the scene
            light_ob.location = pos
            light_ob.rotation_mode = 'QUATERNION'
            light_ob.rotation_quaternion = orient
            # DEfine the properties of the lamp
            light.energy = energy
            light.spot_size = spot_size
            light.spot_blend = spot_blend
            light.shadow_soft_size = shadow_spot_size
            # Add the object to the scene collection
            self.lights.append(light_ob)
            return light_ob

    # Method to add a camera to the scene
    def add_camera(self, pos=(0, 0, 0), orient=(1, 0, 0, 0), obj_distance=None,
                   fstop=0, focal_length=50.0, sensor_size=(8.4594, 7.0932),
                   sensor_px=(2452, 2056), k1=0.0, k2=0.0, k3=0.0,
                   p1=0.0, p2=0.0, p3=0.0, c0=None, c1=None):
        # Create new data and object for the camera
        cam1 = bpy.data.cameras.new("Camera")
        camera = bpy.data.objects.new("Camera", cam1)
        # Introduce the camera to the scene
        bpy.context.collection.objects.link(camera)
        # Define camera properties
        camera.location = pos
        camera.rotation_mode = 'QUATERNION'
        camera.rotation_quaternion = orient
        cam1.lens = focal_length
        cam1.sensor_width = sensor_size[0]
        cam1.sensor_height = sensor_size[1]
        # Add depth of field simulations
        if obj_distance is not None:
            cam1.dof.focus_distance = obj_distance
            cam1.dof.use_dof = True
            cam1.dof.aperture_fstop = fstop
        # Add custom fields and store number of pixels
        # Sensor size
        camera["sensor_px"] = sensor_px
        camera["px_size"] = [i/j for i, j in zip(sensor_size, sensor_px)]
        # Cemera intrinsics
        camera["k1"] = k1
        camera["k2"] = k2
        camera["k3"] = k3
        camera["p1"] = p1
        camera["p2"] = p2
        if c0 is None:
            camera["c0"] = (sensor_px[0])/2
        if c1 is None:
            camera["c1"] = (sensor_px[1])/2
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
        tree = mat.node_tree
        # Remove the default BSDF node
        tree.nodes.remove(tree.nodes["Principled BSDF"])
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
        # Select the target and apply the material
        ob = bpy.context.view_layer.objects.active
        if ob is None:
            bpy.context.view_layer.objects.active = target
        # Assign the material to the cube
        target.data.materials.append(mat)
        target.select_set(True)
        # Enter edit mode
        bpy.ops.object.editmode_toggle()
        # Project the mesh into a cube such that the specimen dimensions
        # fit the encoded physical speckles

        # Find the size of the cube to project to as min(size) / DPI
        small_dim = np.argmin(texImage.image.size) 
        speckle_scaling = texImage.image.size[small_dim] / \
            texImage.image.resolution[small_dim]
        bpy.ops.uv.cube_project(scale_to_bounds=False,
                                correct_aspect=True,
                                cube_size=speckle_scaling)
        bpy.ops.object.editmode_toggle()
        # Add the material to the list
        self.materials.append(mat)
        return mat

    # Define method to set all the properties of the renderer
    def set_renderer(self, cam, n_samples=100, denoising=False):
        scene = bpy.context.scene
        scene.camera = cam
        scene.render.filepath = self.output_path
        scene.render.resolution_x = cam["sensor_px"][0]
        scene.render.resolution_y = cam["sensor_px"][1]
        scene.render.image_settings.file_format = 'TIFF'
        scene.render.image_settings.color_mode = 'BW'
        scene.render.image_settings.color_depth = '8'
        scene.render.image_settings.compression = 0
        scene.render.image_settings.tiff_codec = 'NONE'
        scene.render.engine = 'CYCLES'  # Working
        scene.cycles.device = 'GPU'
        scene.cycles.samples = n_samples
        scene.cycles.use_denoising = denoising
        scene.cycles.denoising_input_passes = 'RGB_ALBEDO'
        scene.render.use_border = False
        scene.render.use_compositing = False

    # Method to calculate rotation between two 3D vectors using Euler angles
    def calc_rot_angle(self, dir1, dir2):
        # Normalise the directions
        dir1 /= np.linalg.norm(dir1) + 1e-16
        dir2 /= np.linalg.norm(dir2) + 1e-16
        # Calculate the axis of rotation
        rot_axis = np.cross(dir1, dir2)
        rot_axis = np.insert(rot_axis, 0, 1 + np.dot(dir1, dir2))
        rot_axis /= np.linalg.norm(rot_axis)
        rot_quat = mathutils.Quaternion(rot_axis)
        return rot_quat

    # Method to multiply two quaternions
    def quaternion_multiply(self, q0, q1):
        w0, x0, y0, z0 = q0
        w1, x1, y1, z1 = q1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0],
                        dtype=np.float64)

    # Method to divide two quaternions
    def quaternion_conjugate(self, q0):
        if type(q0) is list or type(q0) is mathutils.Quaternion:
            q0_conj = [-j if i > 0 else j for i, j in enumerate(q0)]
        elif type(q0) is np.ndarray:
            q0_conj = np.array([-j if i > 0 else j for i, j in enumerate(q0)])
        return q0_conj

    # Method to convert vector to quaternion
    def vec_to_quaternion(self, v):
        if np.linalg.norm(v) == 0:
            w = 1.0
        else:
            w = 0.0
        if type(v) is list:
            v.insert(0, w)
            np.array(v)
        elif type(v) is np.ndarray:
            v = np.insert(v, 0, w)
        elif type(v) is mathutils.Vector:
            v = list(v)
            v.insert(0, w)
            np.array(v)
        return v

    def rotate_vec(self, v, q):
        # Convert the vector to quaternion
        v = self.vec_to_quaternion(v)
        # Calculate the conjugate of rotation quaternion
        q_conj = self.quaternion_conjugate(q)
        # Rotate the vector
        v = self.quaternion_multiply(
            self.quaternion_multiply(q, v), q_conj)
        v = v[1:]
        return v
    
    def rotate_quaternion(self, q0, q1):
        q0_conj = self.quaternion_conjugate(q0)
        q2 = self.quaternion_multiply(
            self.quaternion_multiply(q0, q1), q0_conj)
        return q2

    # Add rotation around z-axis
    def rotate_around_z(self, obj, ang_z):
        # Convert ang from degrees to radians
        ang_z = math.radians(ang_z)
        # Find the current orientation of z-axis
        orient_matrix = obj.matrix_world
        z_axis = [orient_matrix[0][2],
                  orient_matrix[1][2],
                  orient_matrix[2][2]]
        # Retrieve the orientation in quaternions
        q = obj.rotation_quaternion
        # Create rotation vector around the z-axis
        qz = np.array([math.cos(ang_z/2),
                       z_axis[0]*math.sin(ang_z/2),
                       z_axis[1]*math.sin(ang_z/2),
                       z_axis[2]*math.sin(ang_z/2)])
        qz /= np.linalg.norm(qz) + 1e-16
        # Multiply the original orientation with the rotation quaternion
        q2 = self.quaternion_multiply(qz, q)
        # Update the object orientation
        obj.rotation_quaternion = q2

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
            cam1_ang = math.radians(15.0)
            cam1_dist = 1.0
            props["cam1_pos"] = (
                cam1_dist*math.sin(cam1_ang),
                0.0,
                cam1_dist*math.cos(cam1_ang))
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

    # This method imports a *.mesh file that contains information about FE
    # nodes and elements to generate a part in blender. The mesh is then
    # extruded to give some thickness to the part
    def add_FEA_part(self, part_filepath, thickness=0.002):
        # Read the mesh file
        with open(part_filepath, 'r') as file:
            lines = file.readlines()
            # Detect where nodes and elements begin
            tag_lines = [i for i, line in enumerate(lines)
                         if line.startswith('*Node')
                         or line.startswith('*Element')]
            # Define vertices + scale to mm
            # MatchID rotates the mesh by 180 deg
            nodes = list((float(line.split(';')[1]) * (-0.001),
                          float(line.split(';')[2]) * (-0.001),
                          float(line.split(';')[3]) * 0.001)
                         for i, line in enumerate(lines)
                         if not line.startswith('*') and i < tag_lines[1])
            # Define elements
            elements = list((int(line.split(';')[1])-1,
                             int(line.split(';')[2])-1,
                             int(line.split(';')[3])-1,
                             int(line.split(';')[4])-1)
                            for i, line in enumerate(lines)
                            if not line.startswith('*') and i > tag_lines[1])
        # Create mesh
        mesh = bpy.data.meshes.new("part")
        mesh.from_pydata(nodes, [], elements)
        obj = bpy.data.objects.new("specimen", mesh)
        bpy.context.scene.collection.objects.link(obj)
        part = bpy.data.objects["specimen"]
        # Add thickness to the mesh
        part.modifiers.new(name='solidify', type='SOLIDIFY')
        part.modifiers["solidify"].thickness = thickness
        # Return the object
        self.objects.append(part)
        return part

    # This method updates the position of nodes defining the geometry of the
    # FE mesh, allowing to produce images of deformed specimen according to FEM
    def deform_FEA_part(self, part, displ_filepath):
        # Create key shape
        if part.data.shape_keys is None:
            part.shape_key_add()
            # Write the first animation frame
            self.set_new_frame(part)
        sk = part.shape_key_add()
        part.data.shape_keys.use_relative = False
        # Load deformed displacements
        with open(displ_filepath, 'r') as file:
            lines = file.readlines()
            # Detect where nodes and elements begin
            # Define vertices
            nodes = list((float(line.split(';')[1])*0.001,
                          float(line.split(';')[2])*0.001,
                          float(line.split(';')[3])*0.001)
                         for line in lines
                         if not line.startswith('*'))
        # Update the coordinates
        for i in range(len(part.data.vertices)):
            sk.data[i].co = nodes[i]

    # This method adds optical distortions to the rendered image
    def add_image_distortion(self, cam):
        # Define distortion model
        # If no movieclip (needed for distortion model) is present, load a file
        if len(bpy.data.movieclips) == 0:
            bpy.ops.clip.open(files=[{
                              "name": self.pattern_path}],
                              relative_path=False)
        clip = bpy.data.movieclips[0].tracking.camera
        clip.sensor_width = cam.data.sensor_width
        clip.pixel_aspect = cam["sensor_px"][0]/cam["sensor_px"][1]
        clip.focal_length = cam.data.lens
        clip.distortion_model = 'BROWN'
        clip.brown_k1 = cam["k1"]
        clip.brown_k2 = cam["k2"]
        clip.brown_k3 = cam["k3"]
        clip.brown_p1 = cam["p1"]
        clip.brown_p2 = cam["p2"]
        clip.principal[0] = cam["c0"]
        clip.principal[1] = cam["c1"]
        # Build up compositing workflow
        scene = bpy.context.scene
        scene.use_nodes = True
        tree = scene.node_tree
        # Pull the original nodes
        composite_node = tree.nodes["Composite"]
        render_layers_node = tree.nodes["Render Layers"]
        # Remove the original link
        comp_link = composite_node.inputs[0].links[0]
        tree.links.remove(comp_link)
        # Add distortion node
        distortion_node = tree.nodes.new(type="CompositorNodeMovieDistortion")
        distortion_node.clip = bpy.data.movieclips[0]
        distortion_node.distortion_type = 'DISTORT'
        # Add links
        tree.links.new(distortion_node.inputs[0],
                       render_layers_node.outputs[0])
        tree.links.new(distortion_node.outputs[0],
                       composite_node.inputs[0])
        # Enable compositing
        scene.render.use_compositing = True
        # Update the view
        bpy.context.view_layer.update()

    # This method generates the calibration file for stereo DIC in MatchID
    def generate_calib_file(self, cam0, cam1, calib_filepath, ang_mode='XYZ'):
        # Calculate rotation of cam0 to cam1
        cam0_orient = cam0.rotation_quaternion
        cam1_orient = cam1.rotation_quaternion
        # Rotate orientations 180 deg around x axis. Blender and MatchIDs 
        # coordinate systems are not compatible, this rotation corresponds to
        # reflection around x-z and then x-y planes.
        q_x = [math.cos(np.pi/2), math.sin(np.pi/2), 0, 0] # x-axis rotation
        cam0_orient = self.rotate_quaternion(q_x, cam0_orient)
        cam1_orient = self.rotate_quaternion(q_x, cam1_orient)
        # Calculate quaternion rotating one camera to the other
        q_rot = self.quaternion_multiply(
            cam0_orient,
            self.quaternion_conjugate(cam1_orient))
        q_rot_conj = self.quaternion_conjugate(q_rot)
        q_rot = mathutils.Quaternion(q_rot)
        ang = q_rot.to_euler(ang_mode)
        ang = [math.degrees(i) for i in ang]
        # Calculate translation of cam0 to cam1
        dT = (cam0.location - cam1.location) * 1000
        dT[2] *= -1 #reflect z-axis
        dT[1] *= -1 #reflect y-axis
        # Rotate the translation to the cam1 csys
        # First apply rotatation of dT from global to cam0, then cam0->cam1
        dT_rot = self.rotate_vec(self.rotate_vec(dT, cam0_orient), q_rot_conj)
        with open(calib_filepath, 'w') as file:
            file.write('Cam1_Fx [pixels];'
                       + f'{cam0.data.lens/cam0["px_size"][0]}\n')
            file.write('Cam1_Fy [pixels];'
                       + f'{cam0.data.lens/cam0["px_size"][1]}\n')
            file.write('Cam1_Fs [pixels];0\n')
            file.write(f'Cam1_Kappa 1;{cam0["k1"]}\n')
            file.write(f'Cam1_Kappa 2;{cam0["k2"]}\n')
            file.write(f'Cam1_Kappa 3;{cam0["k3"]}\n')
            file.write(f'Cam1_P1;{cam0["p1"]}\n')
            file.write(f'Cam1_P2;{cam0["p2"]}\n')
            file.write(f'Cam1_Cx [pixels];{cam0["c0"]}\n')
            file.write(f'Cam1_Cy [pixels];{cam0["c1"]}\n')
            file.write('Cam2_Fx [pixels];'
                       + f'{cam1.data.lens/cam1["px_size"][0]}\n')
            file.write('Cam2_Fy [pixels];'
                       + f'{cam1.data.lens/cam1["px_size"][1]}\n')
            file.write('Cam2_Fs [pixels];0\n')
            file.write(f'Cam2_Kappa 1;{cam1["k1"]}\n')
            file.write(f'Cam2_Kappa 2;{cam1["k2"]}\n')
            file.write(f'Cam2_Kappa 3;{cam1["k3"]}\n')
            file.write(f'Cam2_P1;{cam1["p1"]}\n')
            file.write(f'Cam2_P2;{cam1["p2"]}\n')
            file.write(f'Cam2_Cx [pixels];{cam1["c0"]}\n')
            file.write(f'Cam2_Cy [pixels];{cam1["c1"]}\n')
            file.write(f'Tx [mm];{dT_rot[0]}\n')
            file.write(f'Ty [mm];{dT_rot[1]}\n')
            file.write(f'Tz [mm];{dT_rot[2]}\n')
            file.write(f'Theta [deg];{ang[0]}\n')
            file.write(f'Phi [deg];{ang[1]}\n')
            file.write(f'Psi [deg];{ang[2]}')

    # Method to enable animation in .blender file
    def set_new_frame(self, obj):
        frame_incr = 20
        # Make the object active
        ob = bpy.context.view_layer.objects.active
        if ob is None:
            bpy.context.view_layer.objects.active = obj
        # Get the current animation frame and increment it
        current_frame = bpy.context.scene.frame_current
        current_frame += frame_incr
        bpy.context.scene.frame_set(current_frame)
        # Insert a new keyframe based on the generated keyposes
        bpy.data.shape_keys["Key"].eval_time = current_frame
        obj.data.shape_keys.keyframe_insert('eval_time',
                                            frame=current_frame)
        bpy.context.scene.frame_end = current_frame
        # bpy.context.view_layer.update()
