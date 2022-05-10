import os
import time

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from rivuletpy.utils.io import loadtiff3d
from rivuletpy.swc import SWC
from rivuletpy.utils.io import loadswc
from rivuletpy.utils.plottools import volume_show, volume_view, flatten
from rivuletpy.utils.filtering import apply_threshold, rolling_ball_removal
from rivuletpy.utils.segmentation import NeuronSegmentor

FORCE = True

if __name__ == '__main__':

    filename = 'data/synthetic-3-cells.tif'

    img = loadtiff3d(filename, out='SITK') # Original Image

    neurons = NeuronSegmentor(img)
    neurons.plot_full_segmentation()

    neuron_images = neurons.neurons

    pass


