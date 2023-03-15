# speckle
A library for creating synthetic speckle patterns for Digital Image Correlation
and rendering realistic images of testing environments.

## Overview and motivation
This library was created out of the need to produce synthetic images that can 
serve as a baseline for image deformation procedure [1] in stereo configuration, 
without having to take reference images, and the corresponding
stereo calibration, in a real setting before being able to perform image
deformation. 

This library enables visualisation and design of novel experimental setups
fully in a virtual setting, which might be useful in many circuimstances, for 
example: 
* Position of the cameras can be optimised even when the testing rig
    is not yet built,
* Novel geometries of full-field based material testing can be explored
    100% virtually,
* Specimen deforms significantly during the experiment so that the change in 
    lighting might be similar 
* Time dependent lighting is used, e.g. such as flash    
    
The library consists of two parts: generation of speckle pattern images and
generation of a virtual experiment using the powerful rendering engine of 
Blender (open-source rending software), controlled via its python API - bpy.

At the moment only one type of speckle pattern is supported, namely the 
optimised speckle pattern (serpentine) proposed by S. Bossuyt [2]. In 
the future, potentially more types will be added.

Although the idea was originally developed out of neccessity for my own 
research, I later came to learn that this has been already explored by
Elizabeth Jones and D. Rohe and published in Experimental Techniques [3], where
they give lots of practical details regarding metrology of virtual experiments,
such as optimal settings for the renderer, expected noise floor due to 
rendering, comparison against MatchID's renderer ImDef engine etc. 

## Installation
To install the software, simply clone the repository and use 
the requirements.txt file to install the neccessary dependencies. Please note that
bpy module is sufficient to produce images with this library, however if one
wanted to inspect/adjust the blender scene in a GUI, a separate blender 
installation is needed. Blender can be found at: https://www.blender.org/

NOTE: Currently using blender 3.4.0, which is compatible with python 3.10.x

# Using the library

For more details regarding the project, please visit the wiki pages:
https://github.com/AleksanderMarek/speckle/wiki that will be gradually 
updated with more information.

## Speckle module
This module generates a random speckle pattern of given image size (in pixels), 
and speckle size (also in pixels). It is possible to set the image such that
it corresponds to certain physical dimensions in millimetres.

More information will be added to the file/wiki later, a basic examples are
included in two examples (example_default_scene.py; example_FEDef.py).

## VirtExp module
This module provides high-level functions to set up a blender scene, where
multiple cameras can look at a target decorated with a speckle pattern.
The image from each camera can be rendered and calibration file linking two
cameras can be generated.

More information will be added to the file/wiki later, a basic examples are
included in two examples (example_default_scene.py; example_FEDef.py).

# References
[1] Lava, P., et al. (2009). "Assessment of measuring errors in DIC using 
deformation fields generated by plastic FEA." Optics and Lasers in 
Engineering 47(7): 747-753.

[2] Bossuyt, S. Optimized patterns for digital image correlation. 
Proceedings of the 2012 Annual Conference on Experimental and 
Applied Mechanics, Imaging Methods for Novel Materials and 
Challenging Applications. 3, 239-248 (2013).

[3] Rohe, D. P. and E. M. C. Jones (2021). "Generation of Synthetic Digital 
Image Correlation Images Using the Open-Source Blender Software." 
Experimental Techniques 46(4): 615-631.

