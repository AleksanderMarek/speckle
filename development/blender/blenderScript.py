import bpy
import math
import random


surf_y_coord = 0
# Spotlight properties
spot_size = math.radians(10+(random.uniform(0, 1)-0.5)*5)
spot_blend = random.uniform(0, 1)
spot_energy = 10.0
X_ang = math.radians(-90+25)
Y_ang = math.radians(45)
R = 0.5+(random.uniform(0, 1)-0.5)*0.2
x = R*math.cos(X_ang)*math.sin(Y_ang)
y = R*math.sin(X_ang)*math.sin(Y_ang)+surf_y_coord
z = R*math.cos(Y_ang)
spot_loc = (x, y, z)
target = (0, surf_y_coord, 0)
rot = (math.atan(-y/z),
       2*math.atan(-x/y),
       math.atan(-x/z))
#rot = (math.radians(55), 0, math.radians(135))       

bpy.data.objects['Spot'].location = spot_loc
bpy.data.objects['Spot'].rotation_euler = rot
spot1 = bpy.data.lights['Spot']
spot1.spot_size = spot_size
spot1.spot_blend = spot_blend
spot1.energy = spot_energy

# Background light properties
# Set light location
light_energy = 1.0
light1 = bpy.data.lights['Light']
light1.energy = light_energy

# Camera properties
# Add camera location
fstop = 6
focal_length = 50
cam1 = bpy.data.cameras['Camera']
cam1.dof.aperture_fstop = fstop
cam1.lens = focal_length

# Add materials to the cube

# Render image and save
output_path = "E:\\GitHub\\speckle\\development\\blender\\test.tiff"
bpy.context.scene.render.filepath = output_path
bpy.ops.render.render(write_still=True)    