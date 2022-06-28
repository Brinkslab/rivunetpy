import os
import time

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from rivunetpy.utils.io import loadtiff3d
from rivunetpy.swc import SWC
from rivunetpy.utils.io import loadswc
from rivunetpy.utils.plottools import volume_show, volume_view, flatten
from rivunetpy.utils.filtering import apply_threshold, rolling_ball_removal
from rivunetpy.utils.segmentation import NeuronSegmentor

FORCE = True

if __name__ == '__main__':

    # filename = r'H:\Duet\Visualizations\MicroscopeImages\488_3_10x_flobj_2msexp_Z_stack-small.tif'
    filename = r"C:\Users\twh\Desktop\Z_stack_small.tif"

    # filename = 'H:\Duet\dataset_s0_c9_4D_20dB.tif'

    img = loadtiff3d(filename, out='SITK') # Original Image

    neurons = NeuronSegmentor(img, tolerance=0.15)
    neurons.plot_full_segmentation()

    neuron_images = neurons.neurons

    pass


