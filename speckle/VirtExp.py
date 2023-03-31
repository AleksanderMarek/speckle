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
    def __init__(
        self,
        pattern_path,
        normal_map_path,
        output_path=None,
        model_path=None,
        objects_position="fixed",
    ):
        """
        Constructor
        """
        # Properties of the scene
        self.pattern_path = pattern_path  # Speckle pattern
        self.normal_map_path = normal_map_path  # Normals map for specularity
        self.output_path = output_path  # Default output path
        self.model_path = model_path  # Path for *.blend path
        self.objects_position = objects_position  # Default position of objects
        # Containers for different objects
        self.objects = list()
        self.lights = list()
        self.cameras = list()
        self.materials = list()
        # Reset the default scene from blender
        bpy.ops.wm.read_factory_settings(use_empty=True)

    # ADDING PARTS TO THE SCENE
    def add_rect_target(self, rect_size):
        """
        The method adds a rectangular object to the scene with the size of
        defined by rect_size. The object is placed at the origin of the
        world. The part faces towards Z direction

        Parameters
        ----------
        rect_size : tuple of len = 3
            Tuple contains size of the target (x, y, z)

        Returns
        -------
        part : bpy.object handle
            bpy.object created by blender

        """
        # Create list of 4 corners
        nodes = [
            (-rect_size[0] / 2, rect_size[1] / 2, 0),
            (-rect_size[0] / 2, -rect_size[1] / 2, 0),
            (rect_size[0] / 2, -rect_size[1] / 2, 0),
            (rect_size[0] / 2, rect_size[1] / 2, 0),
        ]
        # Vector linking nodes to the element
        elements = [(0, 1, 2, 3)]
        thickness = rect_size[2]
        # Create mesh
        mesh = bpy.data.meshes.new("part")
        mesh.from_pydata(nodes, [], elements)
        obj = bpy.data.objects.new("specimen", mesh)
        bpy.context.scene.collection.objects.link(obj)
        part = bpy.data.objects["specimen"]
        # Add thickness to the mesh
        part.modifiers.new(name="solidify", type="SOLIDIFY")
        part.modifiers["solidify"].thickness = thickness
        # Return the object
        self.objects.append(part)
        return part

    def add_FEA_part(
        self,
        part_filepath,
        thickness=0.002,
        solidify_flag=True,
        position=(0, 0, 0),
        rotation=(1, 0, 0, 0),
    ):
        """
        This method imports a *.mesh file that contains information about FE
        nodes and elements to generate a part in blender. The mesh is then
        extruded to give some thickness to the part

        Parameters
        ----------
        part_filepath : str
            Path to *.mesh file
        thickness : float
            DESCRIPTION. The default is 0.002.

        Returns
        -------
        part : bpy.object handle
            bpy.object created by blender

        """
        # Read the mesh file
        with open(part_filepath, "r") as file:
            lines = file.readlines()
            # Detect where nodes and elements begin
            tag_lines = [
                i
                for i, line in enumerate(lines)
                if line.startswith("*Node") or line.startswith("*Element")
            ]
            # Define vertices + scale to mm
            # MatchID rotates the mesh by 180 deg around x-axis
            # TODO: Rotate the coordinate point properly
            nodes = list(
                (
                    float(line.split(";")[1]) * (-0.001),
                    float(line.split(";")[2]) * (-0.001),
                    float(line.split(";")[3]) * 0.001,
                )
                for i, line in enumerate(lines)
                if not line.startswith("*") and i < tag_lines[1]
            )
            # Define elements and offset by 1 (blender starts with i=0)
            # elements is a list of tuples containing node numbers for each
            # mesh element to be created in blender
            elements = list(
                tuple(
                    int(elem_num) - 1
                    for k, elem_num in enumerate(line.split(";"))
                    if k > 0
                )
                for i, line in enumerate(lines)
                if not line.startswith("*") and i > tag_lines[1]
            )
        # Create mesh
        mesh = bpy.data.meshes.new("part")
        mesh.from_pydata(nodes, [], elements)
        obj = bpy.data.objects.new("specimen", mesh)
        bpy.context.scene.collection.objects.link(obj)
        part = bpy.data.objects["specimen"]
        part.rotation_mode = "QUATERNION"
        part["solidify"] = solidify_flag
        part["thickness"] = thickness
        # Add thickness to the mesh
        if part["solidify"]:
            # Select the target and apply the material
            ob = bpy.context.view_layer.objects.active
            if ob is None:
                bpy.context.view_layer.objects.active = obj
            bpy.ops.object.editmode_toggle()
            bpy.ops.mesh.solidify(thickness=thickness)
            bpy.ops.object.editmode_toggle()
            # part.modifiers.new(name='solidify', type='SOLIDIFY')
            # part.modifiers["solidify"].thickness = thickness
        # Move and rotate the object
        part.location = position
        part.rotation_mode = "QUATERNION"
        part.rotation_quaternion = rotation
        # Return the object
        self.objects.append(part)
        return part

    def add_stl_part(self, path, position=(0, 0, 0), rotation=(1, 0, 0, 0), scale=1.0):
        """
        This method imports and *.stl file and adds it to the scene. It
        allows to control the position and orientation of the created part
        and returns a handle to the blender object

        NOTE: Default units of blender are 'm' so the stl should be saved
        with such unit embedded, or alternatively the scale can be set to
        0.001
        """
        # Add the stl model to the scene
        bpy.ops.import_mesh.stl(
            filepath=path,
            global_scale=scale,
            use_scene_unit=False,
            axis_forward="Y",
            axis_up="Z",
        )
        # Get the handle of the imported object
        part = bpy.context.selected_objects[0]
        # Set the position and orientation of the object
        part.location = position
        part.rotation_mode = "QUATERNION"
        part.rotation_quaternion = rotation
        return part

    # ADDING ELEMENTS TO THE SCENE
    def add_light(
        self,
        light_type,
        pos=(0, 0, 0),
        orient=(1, 0, 0, 0),
        energy=0,
        spot_size=0,
        spot_blend=0,
        shadow_spot_size=0,
    ):
        """
        Method to place a new light in the scene

        Parameters
        ----------
        light_type : TYPE
            DESCRIPTION.
        pos : TYPE, optional
            DESCRIPTION. The default is (0, 0, 0).
        orient : TYPE, optional
            DESCRIPTION. The default is (1, 0, 0, 0).
        energy : TYPE, optional
            DESCRIPTION. The default is 0.
        spot_size : TYPE, optional
            DESCRIPTION. The default is 0.
        spot_blend : TYPE, optional
            DESCRIPTION. The default is 0.
        shadow_spot_size : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        light_ob : TYPE
            DESCRIPTION.

        """
        if light_type == "SPOT":
            # Assign properties to the spotlight
            light = bpy.data.lights.new(name="Spot", type="SPOT")
            # Create a new object to represent the light
            light_ob = bpy.data.objects.new(name="Spot", object_data=light)
            # Link the light object to the scene
            bpy.context.collection.objects.link(light_ob)
            # Position the light in the scene
            light_ob.location = pos
            light_ob.rotation_mode = "QUATERNION"
            light_ob.rotation_quaternion = orient
            # DEfine the properties of the lamp
            light.energy = energy
            light.spot_size = spot_size
            light.spot_blend = spot_blend
            light.shadow_soft_size = shadow_spot_size
            # Add the object to the scene collection
            self.lights.append(light_ob)
            return light_ob

    def add_camera(
        self,
        pos=(0, 0, 0),
        orient=(1, 0, 0, 0),
        obj_distance=None,
        fstop=0,
        focal_length=50.0,
        sensor_size=(8.4594, 7.0932),
        sensor_px=(2452, 2056),
        k1=0.0,
        k2=0.0,
        k3=0.0,
        p1=0.0,
        p2=0.0,
        p3=0.0,
        c0=None,
        c1=None,
    ):
        """
        Method to add a camera to the scene

        Parameters
        ----------
        pos : TYPE, optional
            DESCRIPTION. The default is (0, 0, 0).
        orient : TYPE, optional
            DESCRIPTION. The default is (1, 0, 0, 0).
        obj_distance : TYPE, optional
            DESCRIPTION. The default is None.
        fstop : TYPE, optional
            DESCRIPTION. The default is 0.
        focal_length : TYPE, optional
            DESCRIPTION. The default is 50.0.
        sensor_size : TYPE, optional
            DESCRIPTION. The default is (8.4594, 7.0932).
        sensor_px : TYPE, optional
            DESCRIPTION. The default is (2452, 2056).
        k1 : TYPE, optional
            DESCRIPTION. The default is 0.0.

        Returns
        -------
        camera : TYPE
            DESCRIPTION.

        """
        # Create new data and object for the camera
        cam1 = bpy.data.cameras.new("Camera")
        camera = bpy.data.objects.new("Camera", cam1)
        # Introduce the camera to the scene
        bpy.context.collection.objects.link(camera)
        # Define camera properties
        camera.location = pos
        camera.rotation_mode = "QUATERNION"
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
        camera["px_size"] = [i / j for i, j in zip(sensor_size, sensor_px)]
        # Cemera intrinsics
        camera["k1"] = k1
        camera["k2"] = k2
        camera["k3"] = k3
        camera["p1"] = p1
        camera["p2"] = p2
        if c0 is None:
            camera["c0"] = (sensor_px[0]) / 2
        else:
            camera["c0"] = c0
        if c1 is None:
            camera["c1"] = (sensor_px[1]) / 2
        else:
            camera["c1"] = c1
        # Append to the camera list
        self.cameras.append(camera)
        return camera

    def add_material(
        self,
        target,
        glossy_roughness=0.2,
        specular_strength=1.0,
        diffuse_roughness=0.2,
        shader_mix_ratio=0.7,
    ):
        """
        Method to define a new material and add it to the selected object

        Parameters
        ----------
        target : TYPE
            DESCRIPTION.
        glossy_roughness : TYPE, optional
            DESCRIPTION. The default is 0.2.
        specular_strength : TYPE, optional
            DESCRIPTION. The default is 1.0.
        diffuse_roughness : TYPE, optional
            DESCRIPTION. The default is 0.2.
        shader_mix_ratio : TYPE, optional
            DESCRIPTION. The default is 0.7.

        Returns
        -------
        mat : TYPE
            DESCRIPTION.

        """
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
        texImage = tree.nodes.new("ShaderNodeTexImage")
        texImage.location = (-825, 63)
        texImage.image = bpy.data.images.load(self.pattern_path)
        bpy.data.images[0].colorspace_settings.name = "Non-Color"
        tree.links.new(bsdf_glossy.inputs["Color"], texImage.outputs["Color"])
        # Read an image to serve as a normals map for the specular reflection
        norm_image = tree.nodes.new("ShaderNodeTexImage")
        norm_image.location = (-825, 373)
        norm_image.image = bpy.data.images.load(self.normal_map_path)
        bpy.data.images[0].colorspace_settings.name = "Non-Color"
        norm_map = tree.nodes.new("ShaderNodeNormalMap")
        norm_map.location = (-525, 425)
        norm_map.inputs[0].default_value = specular_strength
        norm_map.uv_map = "UVMap"
        tree.links.new(norm_map.inputs["Color"], norm_image.outputs["Color"])
        tree.links.new(bsdf_glossy.inputs["Normal"], norm_map.outputs["Normal"])
        # Add diffusive node (Diffuse BSDF)
        bsdf_diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
        bsdf_diffuse.location = (-424, 21)
        # Set roughness
        bsdf_diffuse.inputs[1].default_value = diffuse_roughness
        # Link the pattern image to the colour of the diffusive reflection
        tree.links.new(bsdf_diffuse.inputs["Color"], texImage.outputs["Color"])
        # Add Mix shader node
        mix_shader = tree.nodes.new("ShaderNodeMixShader")
        mix_shader.location = (42, 340)
        mix_shader.inputs[0].default_value = shader_mix_ratio
        tree.links.new(mix_shader.inputs[1], bsdf_glossy.outputs["BSDF"])
        tree.links.new(mix_shader.inputs[2], bsdf_diffuse.outputs["BSDF"])
        # Link the Mix shader node with the Material output
        mat_output = tree.nodes["Material Output"]
        mat_output.location = (250, 340)
        tree.links.new(mat_output.inputs["Surface"], mix_shader.outputs["Shader"])
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
        # Select all faces
        bpy.ops.mesh.select_all(action="SELECT")
        # Project the mesh into a cube such that the specimen dimensions
        # fit the encoded physical speckles

        # Find the size of the cube to project to as min(size) / DPI
        small_dim = np.argmin(texImage.image.size)
        speckle_scaling = (
            texImage.image.size[small_dim] / texImage.image.resolution[small_dim]
        )
        bpy.ops.uv.cube_project(
            scale_to_bounds=False, correct_aspect=True, cube_size=speckle_scaling
        )
        bpy.ops.object.editmode_toggle()
        # Add the material to the list
        self.materials.append(mat)
        return mat

    # RENDER THE SCENE
    def set_renderer(self, cam, n_samples=200, denoising=False):
        """
        Define method to set all the properties of the renderer
        """
        scene = bpy.context.scene
        scene.camera = cam
        scene.render.filepath = self.output_path
        scene.render.resolution_x = cam["sensor_px"][0]
        scene.render.resolution_y = cam["sensor_px"][1]
        scene.render.image_settings.file_format = "TIFF"
        scene.render.image_settings.color_mode = "BW"
        scene.render.image_settings.color_depth = "8"
        scene.render.image_settings.compression = 0
        scene.render.image_settings.tiff_codec = "NONE"
        scene.render.engine = "CYCLES"  # Working
        scene.cycles.device = "GPU"
        scene.cycles.samples = n_samples
        scene.cycles.use_denoising = denoising
        scene.cycles.denoising_input_passes = "RGB_ALBEDO"
        scene.render.use_border = False
        scene.render.use_compositing = False

    def render_scene(self, filepath=None):
        """
        Method to render the scene and save the output to filepath
        """
        if filepath is not None:
            bpy.context.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)

    def save_model(self):
        """
        Save the blender model to a file
        """
        if self.model_path is not None:
            bpy.ops.wm.save_as_mainfile(filepath=self.model_path)

    def add_image_distortion(self, cam, im_path):
        """
        This method adds optical distortions to the rendered image by taking a
        rendered image and using 'motion tracking' module to add radial and
        tangential distortion based on 7 parameters:
            [k1, k2, k3, p1, p2, c0, c1]
        Within blender this is using compositing and re-rendering the image

        NOTE: At the moment c0 and c1 have to be in the middle of the sensor,
        otherwise the extrinsic parameters will not be right. This is due to
        the fact that change in c0 and c1 affects the effective focal length
        and would require proper calibration routine to establish correct
        values
        """
        # If no movieclip (needed for distortion model) is present, load a file
        # TODO: Change the clip every time (so that each image can be
        # distorted properly)
        if len(bpy.data.movieclips) == 0:
            bpy.ops.clip.open(files=[{"name": im_path}], relative_path=False)
        # Apply parameters to the distortion model
        clip = bpy.data.movieclips[0].tracking.camera
        clip.sensor_width = cam.data.sensor_width
        clip.pixel_aspect = cam["sensor_px"][0] / cam["sensor_px"][1]
        clip.focal_length = cam.data.lens
        clip.distortion_model = "BROWN"
        clip.brown_k1 = cam["k1"]
        clip.brown_k2 = cam["k2"]
        clip.brown_k3 = cam["k3"]
        clip.brown_p1 = cam["p2"]  # MatchID permutes p1/p2 compared to blender
        clip.brown_p2 = -cam["p1"]
        clip.principal[0] = cam["c0"]
        clip.principal[1] = cam["c1"]
        # Build up compositing workflow
        scene = bpy.context.scene
        scene.use_nodes = True
        tree = scene.node_tree
        # Pull the original nodes
        composite_node = tree.nodes["Composite"]
        # Remove the original link
        comp_link = composite_node.inputs[0].links[0]
        tree.links.remove(comp_link)
        # Add image node
        im_node = tree.nodes.new(type="CompositorNodeImage")
        im_node.image = bpy.data.images.load(im_path)
        # Add distortion node
        distortion_node = tree.nodes.new(type="CompositorNodeMovieDistortion")
        distortion_node.clip = bpy.data.movieclips[0]
        distortion_node.distortion_type = "DISTORT"
        # Add links
        tree.links.new(distortion_node.inputs[0], im_node.outputs[0])
        tree.links.new(distortion_node.outputs[0], composite_node.inputs[0])
        # Enable compositing
        scene.render.use_compositing = True
        # Update the view
        bpy.context.view_layer.update()
        # Render the distorted image
        self.render_scene(im_path)
        # Disable compositing
        scene.render.use_compositing = False

    # METHODS TO INTEGRATE WITH MATCHID
    def generate_calib_file(self, cam0, cam1, calib_filepath, ang_mode="XYZ"):
        """
        This method generates the calibration file for stereo DIC in MatchID
        """
        # Calculate rotation of cam0 to cam1 by getting orientation of both
        # cameras and using quaternion division to get the rotation. Then
        # the resulting quaternion is converted to Euler angles
        cam0_orient = cam0.rotation_quaternion
        cam1_orient = cam1.rotation_quaternion
        # Rotate orientations 180 deg around x axis. Blender and MatchIDs
        # coordinate systems are not compatible, this rotation corresponds to
        # reflection around x-z and then x-y planes.
        q_x = [math.cos(np.pi / 2), math.sin(np.pi / 2), 0, 0]  # x-axis rotation
        cam0_orient = self.rotate_quaternion(q_x, cam0_orient)
        cam1_orient = self.rotate_quaternion(q_x, cam1_orient)
        # Calculate quaternion rotating one camera to the other
        q_rot = self.quaternion_multiply(
            cam0_orient, self.quaternion_conjugate(cam1_orient)
        )
        q_rot_conj = self.quaternion_conjugate(q_rot)
        q_rot = mathutils.Quaternion(q_rot)
        ang = q_rot.to_euler(ang_mode)
        ang = [math.degrees(i) for i in ang]
        # Calculate translation of cam0 to cam1 and rotate the vector
        # to the orientation of cam1
        dT = (cam0.location - cam1.location) * 1000
        dT[2] *= -1  # reflect z-axis
        dT[1] *= -1  # reflect y-axis
        # Rotate the translation to the cam1 csys
        # First apply rotatation of dT from global to cam0, then cam0->cam1
        dT_rot = self.rotate_vec(self.rotate_vec(dT, cam0_orient), q_rot_conj)
        # Create *.caldat file
        with open(calib_filepath, "w") as file:
            file.write("Cam1_Fx [pixels];" + f'{cam0.data.lens/cam0["px_size"][0]}\n')
            file.write("Cam1_Fy [pixels];" + f'{cam0.data.lens/cam0["px_size"][1]}\n')
            file.write("Cam1_Fs [pixels];0\n")
            file.write(f'Cam1_Kappa 1;{cam0["k1"]}\n')
            file.write(f'Cam1_Kappa 2;{cam0["k2"]}\n')
            file.write(f'Cam1_Kappa 3;{cam0["k3"]}\n')
            file.write(f'Cam1_P1;{cam0["p1"]}\n')
            file.write(f'Cam1_P2;{cam0["p2"]}\n')
            file.write(f'Cam1_Cx [pixels];{cam0["c0"]}\n')
            file.write(f'Cam1_Cy [pixels];{cam0["c1"]}\n')
            file.write("Cam2_Fx [pixels];" + f'{cam1.data.lens/cam1["px_size"][0]}\n')
            file.write("Cam2_Fy [pixels];" + f'{cam1.data.lens/cam1["px_size"][1]}\n')
            file.write("Cam2_Fs [pixels];0\n")
            file.write(f'Cam2_Kappa 1;{cam1["k1"]}\n')
            file.write(f'Cam2_Kappa 2;{cam1["k2"]}\n')
            file.write(f'Cam2_Kappa 3;{cam1["k3"]}\n')
            file.write(f'Cam2_P1;{cam1["p1"]}\n')
            file.write(f'Cam2_P2;{cam1["p2"]}\n')
            file.write(f'Cam2_Cx [pixels];{cam1["c0"]}\n')
            file.write(f'Cam2_Cy [pixels];{cam1["c1"]}\n')
            file.write(f"Tx [mm];{dT_rot[0]}\n")
            file.write(f"Ty [mm];{dT_rot[1]}\n")
            file.write(f"Tz [mm];{dT_rot[2]}\n")
            file.write(f"Theta [deg];{ang[0]}\n")
            file.write(f"Phi [deg];{ang[1]}\n")
            file.write(f"Psi [deg];{ang[2]}")

    def deform_FEA_part(self, part, displ_filepath):
        """
        This method updates the position of nodes defining the geometry of the
        FE mesh, allowing to produce images of deformed specimen according to
        FEM.

        The approach is to use shape keys so that it is possible to animate
        the test in blender using key frames for each key shape

        Mind that basic unit in blender is m, therefore mm coordinates have to
        be multiplied by 0.001
        """
        # Create key shape
        if part.data.shape_keys is None:
            part.shape_key_add()
            # Write the first animation frame
            self.set_new_frame(part)
        sk = part.shape_key_add()
        part.data.shape_keys.use_relative = False
        # Load deformed displacements
        with open(displ_filepath, "r") as file:
            lines = file.readlines()
            # Detect where nodes and elements begin
            # Define vertices
            nodes = list(
                (
                    float(line.split(";")[1]) * 0.001,
                    float(line.split(";")[2]) * 0.001,
                    float(line.split(";")[3]) * 0.001,
                )
                for line in lines
                if not line.startswith("*")
            )
        # Update the coordinates
        # Save coordinates of the original layer of nodes
        # TODO: SOLIDIFY Deformation will only work for
        # 2D specimen at the moment
        if part["solidify"]:
            n_nodes_layer = int(len(part.data.vertices) / 2)
        else:
            n_nodes_layer = int(len(part.data.vertices))
        all_nodes = np.array([sk.data[i].co for i in range(len(part.data.vertices))])
        first_layer = all_nodes[0:n_nodes_layer, :]
        # Currently faking the thickness of specimen so the back plane just
        # follows the front plane rigidly
        for i in range(len(part.data.vertices)):
            if i < n_nodes_layer:  # Original plane
                sk.data[i].co = nodes[i]
            else:  # Solidified plane
                # Find the closest node on the original plane
                dist = np.linalg.norm(first_layer - sk.data[i].co, axis=1)
                cn = np.argmin(dist)
                # Apply the coordinates from the original layer
                sk.data[i].co = nodes[cn]
                # Correct for the thickness
                sk.data[i].co[2] -= part["thickness"]

    def set_new_frame(self, obj):
        """
        Method to enable animation in .blender file by creating key frames
        from key shapes
        """
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
        obj.data.shape_keys.keyframe_insert("eval_time", frame=current_frame)
        bpy.context.scene.frame_end = current_frame

    # DEFAULT SETTINGS
    def create_def_scene(self):
        """
        This method creates a demonstrator scene with two cameras:
            cam0 - perpendicular to the rectangular target
            cam1 - Cross camera at 15 degrees to the target
            spotlight
        """
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
        light_angle = self.calc_rot_angle(p["light_init_rot"], light_target_orient)
        self.add_light(
            p["light_type"],
            pos=p["light_pos"],
            orient=light_angle,
            energy=p["light_energy"],
            spot_size=p["light_spotsize"],
            spot_blend=p["light_spot_blend"],
            shadow_spot_size=p["light_shad_spot"],
        )
        # Add straight camera
        # Calculate desired orientation of the cam
        cam0_target_orient = p["cam0_target"] - np.array(p["cam0_pos"])
        cam0_target_dist = np.linalg.norm(cam0_target_orient) + 1e-16
        cam_angle = self.calc_rot_angle(p["cam_init_rot"], cam0_target_orient)
        cam0 = self.add_camera(
            pos=p["cam0_pos"],
            orient=cam_angle,
            fstop=p["cam_fstop"],
            focal_length=p["cam_foc_length"],
            obj_distance=cam0_target_dist,
        )
        # Add cross camera
        cam1_target_orient = p["cam1_target"] - np.array(p["cam1_pos"])
        cam1_target_dist = np.linalg.norm(cam1_target_orient) + 1e-16
        cam_angle = self.calc_rot_angle(p["cam_init_rot"], cam1_target_orient)
        self.add_camera(
            pos=p["cam1_pos"],
            orient=cam_angle,
            fstop=p["cam_fstop"],
            focal_length=p["cam_foc_length"],
            obj_distance=cam1_target_dist,
        )
        # Define the material and assign it to the cube
        self.add_material(
            target,
            glossy_roughness=p["glossy_roughness"],
            specular_strength=p["specular_strength"],
            diffuse_roughness=p["diffuse_roughness"],
            shader_mix_ratio=p["shader_mix"],
        )
        # Set the renderer up and render image
        self.set_renderer(cam0)
        # Save the model
        self.save_model()

    def get_default_params(self):
        """
        This method contains parameters (such as positions, materials,
        intensity etc.) for constructing the default scene. There are two modes:
        fixed, where the scene is set manually (as decribed in method
        create_def_scene). The other mode 'random' randomises the positions and
        object/material properties to produce varied scene every time it is reset
        """
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
        props["glossy_roughness"] = 0.2
        props["diffuse_roughness"] = 0.5
        # Define parameters that are allowed to change
        # Constant
        if self.objects_position == "fixed":
            props["light_pos"] = (-0.5, 0.3, 0.5)
            props["light_target"] = np.array([0.0, 0.0, 0.0])
            props["light_energy"] = 20.0
            props["light_spotsize"] = math.radians(25)
            props["light_spot_blend"] = 0.5
            props["light_shad_spot"] = 0.01
            cam1_ang = math.radians(15.0)
            cam1_dist = 1.0
            props["cam1_pos"] = (
                cam1_dist * math.sin(cam1_ang),
                0.0,
                cam1_dist * math.cos(cam1_ang),
            )
            props["cam_fstop"] = 8.0
            props["cam1_target"] = np.array([0.0, 0.0, 0.0])
            props["specular_strength"] = 1.0
            props["shader_mix"] = 0.7
        # Random
        elif self.objects_position == "random":
            # Control parameters
            polar_ang_variation = 75
            azim_ang_variation = 45
            spot_dist_mean = 1.0
            spot_dist_variation = 0.5
            target_range = 0.075
            spot_energy_mean = 17.5
            spot_energy_variation = 5.0
            spot_ang_mean = 45.0
            spot_ang_variation = 15.0
            spot_size_min = 0.0
            spot_size_max = 0.01
            cam1_dist_mean = 1.0
            cam1_dist_variation = 0.005
            cam1_ang_mean = 15.0
            cam1_ang_variation = 5.0
            cam1_target_variation = 0.03
            fstop_min = 4.0
            fstop_max = 11.0
            specular_strength_mean = 0.3
            specular_strength_std = 0.2
            shader_mix_ratio_min = 0.5
            shader_mix_ratio_max = 0.95
            # Position of the spotlight
            polar_ang = math.radians(random.uniform(-1, 1) * polar_ang_variation)
            azim_ang = math.radians(random.uniform(-1, 1) * azim_ang_variation)
            spot_dist = spot_dist_mean + random.uniform(-1, 1) * spot_dist_variation
            x = spot_dist * math.cos(polar_ang) * math.sin(azim_ang)
            y = spot_dist * math.sin(polar_ang) * math.sin(azim_ang)
            z = spot_dist * math.cos(azim_ang)
            props["light_pos"] = (x, y, z)
            # Light target
            props["light_target"] = (
                random.uniform(-target_range, target_range),
                0,
                random.uniform(-target_range, target_range),
            )
            # Light energy
            light_energy = random.normalvariate(spot_energy_mean, spot_energy_variation)
            while light_energy <= 0:
                light_energy = random.normalvariate(1, 0.2)
            props["light_energy"] = light_energy

            # Spot size
            props["light_spotsize"] = math.radians(
                spot_ang_mean + random.uniform(-1, 1) * spot_ang_variation
            )
            # Spot blend
            props["light_spot_blend"] = random.uniform(0.0, 1.0)
            # Light shadow spot
            props["light_shad_spot"] = random.uniform(spot_size_min, spot_size_max)
            # Position of cross camera
            cam1_dist = random.normalvariate(cam1_dist_mean, cam1_dist_variation)
            cam1_ang = math.radians(
                random.normalvariate(cam1_ang_mean, cam1_ang_variation)
            )
            props["cam1_pos"] = (
                cam1_dist * math.sin(cam1_ang),
                0.0,
                cam1_dist * math.cos(cam1_ang),
            )
            # Cross camera target
            props["cam1_target"] = (
                random.normalvariate(0.0, cam1_target_variation),
                random.normalvariate(0.0, cam1_target_variation),
                0.0,
            )
            # Aperture
            props["cam_fstop"] = random.uniform(fstop_min, fstop_max)
            # Material properties
            spec_strength = random.normalvariate(
                specular_strength_mean, specular_strength_std
            )
            shader_mix = random.uniform(shader_mix_ratio_min, shader_mix_ratio_max)
            props["specular_strength"] = spec_strength
            props["shader_mix"] = shader_mix
        return props

    # HELPER FUNCTIONS

    def calc_rot_angle(self, dir1, dir2):
        """
        Method to calculate rotation between two 3D vectors using Euler angles.
        In short, first a rotation axis is found by cross product, then
        the rotation angle is found using dot product and the result is output
        as quaternion [q, i, j, k]
        """
        # Normalise the directions
        dir1 /= np.linalg.norm(dir1) + 1e-16
        dir2 /= np.linalg.norm(dir2) + 1e-16
        # Calculate the axis of rotation
        rot_axis = np.cross(dir1, dir2)
        # Calculate rotation and insert as real part of quaternion
        rot_axis = np.insert(rot_axis, 0, 1 + np.dot(dir1, dir2))
        # Normalise the quaternion
        rot_axis /= np.linalg.norm(rot_axis)
        # Orientation vector will be colinear with the desired direction but
        # It might be poiting the opposite way. Check it with the dot product
        rotated_dir1 = self.rotate_vec(dir1, self.quaternion_conjugate(rot_axis))
        # Check if the dp is (-1) with some tolerance
        if np.abs(np.dot(rotated_dir1, dir2) + 1) < 1e-3:
            # Add another 180 deg rotation to the quaternion
            # This is in quaternion: [0, x, y, z] of the original rotation quat
            rot_axis = self.quaternion_multiply(
                rot_axis, np.array(np.concatenate(([0], rot_axis[1:]), axis=0))
            )
            # If the direction coincides with z axis and the quaternion zeros
            # then replace it with a 180 deg rotation around x-axis
            if np.linalg.norm(rot_axis) == 0:
                rot_axis = np.array([0, 1, 0, 0])
        rot_quat = mathutils.Quaternion(rot_axis)

        return rot_quat

    def quaternion_multiply(self, q0, q1):
        """
        Method to multiply two quaternions

        TODO: convert to quaternion type?
        """
        w0, x0, y0, z0 = q0
        w1, x1, y1, z1 = q1
        return np.array(
            [
                -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            ],
            dtype=np.float64,
        )

    def quaternion_conjugate(self, q0):
        """
        Method to calculate the conjugate of a quaternion
        TODO: Standardise quaternion variable type across the code
        """
        if type(q0) is list or type(q0) is mathutils.Quaternion:
            q0_conj = [-j if i > 0 else j for i, j in enumerate(q0)]
        elif type(q0) is np.ndarray:
            q0_conj = np.array([-j if i > 0 else j for i, j in enumerate(q0)])
        return q0_conj

    def vec_to_quaternion(self, v):
        """
        Method to convert vector to quaternion by inserting 0 as the first
        element

        TODO: Standardise vector type across code
        """
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
        """
        This method rotates vector v by quaternion q according to formula:
            v = q.v.conj(q)
        """
        # Convert the vector to quaternion
        v = self.vec_to_quaternion(v)
        # Calculate the conjugate of rotation quaternion
        q_conj = self.quaternion_conjugate(q)
        # Rotate the vector
        v = self.quaternion_multiply(self.quaternion_multiply(q, v), q_conj)
        # Output vector is the imaginary part of the quaternion
        v = v[1:]
        return v

    def rotate_quaternion(self, q0, q1):
        """Rotate quaternion q0 by quaternion q1 according to the formula:
        q0 = q1.q0.conj(q0)
        """
        q0_conj = self.quaternion_conjugate(q0)
        q2 = self.quaternion_multiply(self.quaternion_multiply(q0, q1), q0_conj)
        return q2

    def rotate_around_z(self, obj, ang_z):
        """
        Method to rotate the object obj, around its local z-axis by angle ang_z
        This is done by pulling orientation matrix matrix_world from blender,
        identifying the orientation of the z-axis and using formula
        for quaternion defying rotation around an axis to produce new
        orientation
        """
        # Convert ang from degrees to radians
        ang_z = math.radians(ang_z)
        # Find the current orientation of z-axis
        orient_matrix = obj.matrix_world
        z_axis = [orient_matrix[0][2], orient_matrix[1][2], orient_matrix[2][2]]
        # Retrieve the orientation in quaternions
        q = obj.rotation_quaternion
        # Create rotation vector around the z-axis
        qz = np.array(
            [
                math.cos(ang_z / 2),
                z_axis[0] * math.sin(ang_z / 2),
                z_axis[1] * math.sin(ang_z / 2),
                z_axis[2] * math.sin(ang_z / 2),
            ]
        )
        # Normalise the quaternion
        qz /= np.linalg.norm(qz) + 1e-16
        # Multiply the original orientation with the rotation quaternion
        q2 = self.quaternion_multiply(qz, q)
        # Update the object orientation
        obj.rotation_quaternion = q2

    # Method to remove an object from the scene
    def remove_object(self, obj):
        bpy.data.objects.remove(obj, do_unlink=True, do_id_user=True, do_ui_user=True)
