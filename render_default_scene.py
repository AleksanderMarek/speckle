"""
This script generates the default scene and renders two images.
The scene consists of a plate target and two cameras, the first one
is positioned perpendicularly to the target while the other is positioned
at an angle of 15 degrees to the target
"""

from speckle import VirtExp


pattern_path = r"E:\speckle\test\im.tiff"
normals_path = r"E:\speckle\test\grad.tiff"
model_path = r"E:\speckle\test\model_dev.blend"
output_path = r"E:\speckle\test\render_0.tiff"
output_path2 = r"E:\speckle\test\render_1.tiff"

a = VirtExp(pattern_path, normals_path, output_path, model_path,
            objects_position="fixed")
# Set up the default scene
a.create_def_scene()  
# Render the scene with the perpendicular camera
a.render_scene()
# Switch the camera to the cross one and render the scene
a.set_renderer(a.cameras[1])
a.render_scene(output_path2) 

