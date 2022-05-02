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

    plt.style.use('dark_background')

    filename = 'data/Synthetic-no-bg.tif'

    img = loadtiff3d(filename, out='SITK') # Original Image

    plt.imshow(flatten(img), cmap='gray')
    plt.title(img.GetPixelID())
    plt.colorbar()
    plt.show()

    neurons = NeuronSegmentor(img, save=True)
    neurons.plot_full_segmentation()

    neuron_images = neurons.neuron_images

    for image in neuron_images:
        plt.imshow(flatten(image), cmap='gray')
        plt.title(img.GetPixelID())
        plt.colorbar()
        plt.show()

    pass


