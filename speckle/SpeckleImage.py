""" This library is designed to produce synthetic speckle patterns that
can be used to quantify metrological performance of digital imge 
correlation (DIC) setups


TODO:
"""

import math
import numpy as np
import numpy.matlib 
from PIL import Image
from matplotlib import pyplot as plt
from scipy import ndimage

# Define class SpeckleImage that produces a bitmap corresponding to a random
# speckle pattern with defined properties
class SpeckleImage:
    # Constructor
    def __init__(self, image_size, speckle_size, algorithm='optim', 
                 DPI=450, binary_high=1.0, binary_low=0.0, noise_mag=None,
                 normal_map_filter=3):
        image_size = tuple(tuple(map(int, image_size)))
        self.image_size = image_size
        self.speckle_size = speckle_size
        self.algorithm = algorithm
        self.pattern = None
        self.DPI = DPI
        self.binarised_high = binary_high
        self.binarised_low = binary_low
        self.noise_mag = noise_mag
        self.normal_map_filter = normal_map_filter
        
    # Generate raw speckle pattern given properties
    def gen_pattern(self):
        # This function generates a synthetic pattern according to the
        # specified algorithm
        if self.algorithm == 'optim':
            self.optimised_pattern()
        return self.pattern
    
    # Algorithm for generating optimised speckle pattern developed by
    # S. Bossuyd
    def optimised_pattern(self):
        """ This algorithm generates a random speckle pattern by 
        creating a ring in Fourier domain at a certain distance from
        DC component and using inverse FFT to produce the image.
        The radius of the ring controls the speckle size but the
        original fourier canvas has to be oversized as otherwise
        artefacts are produced in the corners of the image. The image
        is then binarized to produce black and white speckle image
        """
        fft_threshold = 8 # Thickness of the FFT ring
        # Produce oversized canvas and calculate distance of each pixel
        # to the centre of it
        canvas_size = int(1.5*max(self.image_size))
        canvas = np.zeros([canvas_size, canvas_size], dtype = 'complex')
        canvas_centre = math.floor(canvas_size)/2
        X, Y = np.meshgrid(range(canvas_size), range(canvas_size))
        dist = np.sqrt(np.power(X-canvas_centre,2)+
                       np.power(Y-canvas_centre,2))
        # Calculate the radius of the ring given the desired speckle size
        # factor of 0.5 comes from the fact that each wavelength is two 
        # speckles (one black and one white)
        fft_radius = 0.5*math.floor(canvas_size/self.speckle_size)
        # Construct the ring in the fourier space and use inverse FFT 
        # to produce the image. Add random element to the phase of the 
        # FFT ring (imaginary part)
        fft_ring = np.argwhere(np.abs(dist-fft_radius)<=fft_threshold)
        canvas[fft_ring[:,0], fft_ring[:,1]] = 1.0
        canvas[fft_ring[:,0], fft_ring[:,1]] \
            += np.random.rand(fft_ring.shape[0])*1j*2*math.pi
        speckle_im = np.fft.ifft2(np.fft.ifftshift(canvas))
        speckle_im = np.real(speckle_im)
        # Crop the image to the desired size
        y_crop_start = int(canvas_centre-math.floor(self.image_size[1]/2))
        y_crop_end = y_crop_start+self.image_size[1]
        x_crop_start = int(canvas_centre-math.floor(self.image_size[0]/2))
        x_crop_end = x_crop_start+self.image_size[0]
        speckle_im = speckle_im[y_crop_start:y_crop_end,
                                x_crop_start:x_crop_end]
        # Adjust histogram and binarize the result
        speckle_im = (speckle_im-np.min(speckle_im)) \
                     / (np.max(speckle_im)-np.min(speckle_im))
        self.pattern = (np.where(speckle_im > 0.5,
                                self.binarised_high, 
                                self.binarised_low)*255).astype('uint8')
        if self.noise_mag is not None:
            self.pattern += np.random.normal(
                                0, 
                                self.noise_mag, 
                                size=self.pattern.shape).astype('uint8')

    # Plots the image of the pattern
    def im_show(self):
        plt.imshow(self.pattern, cmap='gray')
        plt.show()
        
    # Save the generated speckle pattern as an image
    def im_save(self, path):
        im = Image.fromarray(self.pattern, mode='L')
        im.save(path)
        
    # Set canvas size, speckle size and DPI based on physical dimensions
    def set_physical_dim(self, physical_size, speckle_size, DPI):
        # Adjust image_size based on the desired physical size and DPI
        image_size = (math.ceil(physical_size[0]/25.4*DPI),
                      math.ceil(physical_size[1]/25.4*DPI))
        self.image_size = image_size
        speckle_size_px = speckle_size/25.4*DPI
        self.speckle_size = speckle_size_px
        self.DPI = DPI
        
    # Calculate gradient of the pattern
    def pattern_gradient(self):
        # Get x-gradient in "sx"
        sx = ndimage.sobel(self.pattern,axis=0,mode='constant')
        # Get y-gradient in "sy"
        sy = ndimage.sobel(self.pattern,axis=1,mode='constant')
        # Get square root of sum of squares
        self.gradient = np.hypot(sx,sy)
        self.gradient = (np.where(self.gradient > 128,
                                self.binarised_high, 
                                self.binarised_low)*255).astype('uint8')
        
    # Save gradient image
    def grad_save(self, path):
        im = Image.fromarray(self.norm_map, mode='RGB')
        im.save(path)
    
    # Generate normals map based on the gradient pattern    
    def generate_norm_map(self, binary_map=None):
        # TODO: Add a possibility of relating the normals map to the 
        # gradient of a speckle pattern
        mat_one = np.ones((self.pattern.shape[0], self.pattern.shape[1], 1))
        # R channel
        mean_mat = mat_one * 127
        std_mat = mat_one * 30 \
            + np.multiply(mat_one, self.pattern[:, :, None])*30
        R = np.random.normal(mean_mat, std_mat).astype('uint8')    
        # G channel    
        mean_mat = mat_one * 127
        std_mat = mat_one * 30 \
            + np.multiply(mat_one, self.pattern[:, :, None])*30
        G = np.random.normal(mean_mat, std_mat).astype('uint8')    
        # B channel
        B = np.random.uniform(0, 255, mat_one.shape).astype('uint8')     
        self.norm_map = np.dstack([R, G, B])
        self.norm_map = ndimage.median_filter(self.norm_map,
                                              self.normal_map_filter)