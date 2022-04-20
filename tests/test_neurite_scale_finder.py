import os
import time

import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from mpl_toolkits.axes_grid1.inset_locator import (InsetPosition)
import SimpleITK as sitk

from rivuletpy.utils.io import loadtiff3d
from rivuletpy.utils.plottools import volume_show, volume_view, flatten
from rivuletpy.utils.filtering import apply_threshold, rolling_ball_removal
from rivuletpy.utils.segmentation import NeuronSegmentor

plt.style.use('dark_background')

filename = 'data/Synthetic-no-bg.tif'

fig = plt.figure(figsize=(10, 5))

img = loadtiff3d(filename, out='SITK') # Original Image

binary, threshold = apply_threshold(img)
#
# start = time.time()

neurons = NeuronSegmentor(img)

print(neurons)

pass


