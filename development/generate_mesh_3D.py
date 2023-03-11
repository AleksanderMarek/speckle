from abaqus import*
from abaqusConstants import*
import visualization
import numpy as np

"""
This script generates *.mesh file for a model in Abaqus using 3D elements.
It is very crude and number of assumptions were made to simplify things. 
1. The model consists only of brick elements (C3D8R in a model tested). 
	It WILL NOT work if any tetrahedral elements are present (but should be easy
	to update)
2. A surface is created in Abaqus spanning the entire model. When defining model
	go to Assembly -> Surfaces -> Select the entire model. 
	NOTE: At the moment the script will only work if there's only one surface
	defined. If there are others, a simple fix would be to change 
	desired_surf to the key (name) used in Abaqus, e.g. 'SURF-1'
3. In a preliminary test it seemed that the surface object returns all mesh 
	faces and corresponding elements (if the same element has 3 faces included
	in the surface definition, it will be returned 3 times, each time with 
	different face). This is not confirmed so might not always work
4. Face numbering was taken from http://dsk-016-1.fsid.cvut.cz:2080/v6.12/books/usb/default.htm?startat=pt06ch28s01alm01.html
5. Note: Some +/- 1 in index references are due to Python indexing from 0 and 
	Abaqus from 1
"""

odb_path = r'C:\local\T_joint_3D.odb'
output_path = r'D:\Experiment Quality\Correlation test\T_joint\T_joint_3D.mesh'

# Load ODB path
odb = session.openOdb(
    name=odb_path)
	
# Get the surfaces object	
surfaces = odb.rootAssembly.surfaces

# List of all nodes belonging to the surface
nodes = []
n_nodes = int(len(odb.rootAssembly.instances['PART-1-1'].nodes))+1
nodes_coords = np.array([i.coordinates for i in \
	odb.rootAssembly.instances['PART-1-1'].nodes])

def element_2D_connectivity(connectivity, face):
	"""
	This function takes the connectivity data of a 3D element and its face
	and returns the node numbers assigned to the face
	"""
	mesh = []
	if face == FACE1:
		mesh = [0, 1, 2, 3]
	elif face == FACE2:
		mesh = [4, 5, 6, 7]
	elif face == FACE3:
		mesh = [0, 1, 5, 4]
	elif face == FACE4:
		mesh = [1, 2, 6, 5]
	elif face == FACE5:
		mesh = [2, 3, 7, 6]
	elif face == FACE6:
		mesh = [3, 0, 4, 7]
	nodes = [connectivity[i] for i in mesh]
	return nodes

desired_surf = surfaces.keys()
for surf in desired_surf: # Loop over all surfaces
	# Get list of nodes, elements and faces (there are as many entries as there
	# are faces in the surface
	nodes_list = odb.rootAssembly.surfaces[surf].nodes[0]
	elem_list = odb.rootAssembly.surfaces[surf].elements[0]
	face_list = odb.rootAssembly.surfaces[surf].faces[0]
	# For every face fetch the labels of all connected nodes and 
	# remove duplicates
	#[nodes.append(i.label) for i in nodes_list] # old version
	nodes = np.unique(np.array([i.label for i in nodes_list]))
	# Create mapping between Abaqus numbering and the new node numbering
	# including only surface nodes
	nodes_mapping = np.empty((n_nodes,), dtype=np.int32)
	for i, j in enumerate(nodes):
		nodes_mapping[j] = int(i+1)
	# Get coordinates of the surface nodes	
	#surf_nodes = np.array([i.coordinates for i in nodes_list]) #old version
	# Pull coordinates from the list of all nodes based on the label
	surf_nodes = np.array([nodes_coords[i-1] for i in nodes])
	# Build up surface elements using faces data
	elems = []
	for i in range(len(elem_list)):
		elems.append(element_2D_connectivity(elem_list[i].connectivity,
											 face_list[i]))

# Write the initial coordinates (surf_nodes) and surface elements (elems) to 
# the *.mesh file
with open(output_path, 'w') as file:
		file.write('*Part, name=Part-1\n')
		file.write('*Node\n')
		# Write number and coordinates of all surface nodes. Make sure to 
		# start numbering nodes with 1
		[file.write(str(i+1) + ';' +
					str(j[0]) + ';' +
					str(j[1]) + ';' +
					str(j[2]) +
					'\n') for i, j in enumerate(surf_nodes)]
		file.write('*Element\n')
		# Write number of the element and associated node numbers
		# TODO: Change to accomodate for different number of nodes in the face
		[file.write(str(i+1) + ';' + 
					str(nodes_mapping[j[0]]) + ';' +
					str(nodes_mapping[j[1]]) + ';' +
					str(nodes_mapping[j[2]]) + ';' +
					str(nodes_mapping[j[3]]) + 
					'\n') for i, j in enumerate(elems)]