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

    filename = 'data/AVG_dataset_s0_c9-SNR_25-3.tif'

    img = loadtiff3d(filename, out='SITK') # Original Image

    neurons = NeuronSegmentor(img)
    neurons.plot_full_segmentation()

    neuron_images = neurons.neurons

    pass


