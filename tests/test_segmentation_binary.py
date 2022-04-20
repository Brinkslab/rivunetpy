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
from rivuletpy.utils.segmentation import get_soma_scale

plt.style.use('dark_background')

# CAMERA = {'pos': 2, 'foc': 1, 'az': 30, 'el': 45}
# WINDOW = {'w': 600, 'h': 1200}

filename = 'data/Synthetic-no-bg.tif'

# fig, axs = plt.subplots(2, 3, dpi=300)
# fig.suptitle(f'{threshold_method}', fontsize=12)

# imshow_flatten(axs[0, 0], img, cmap='gray')
# axs[0, 0].set_title('Original Image')
# axs[0, 0].set_axis_off()
#
# imshow_flatten(axs[0, 1], binary, cmap='gray')
# axs[0, 1].set_title(f'Binary Image ({threshold_method})')
# axs[0, 1].set_axis_off()

fig = plt.figure(figsize=(10, 5), dpi=300)

img = loadtiff3d(filename, out='SITK') # Original Image

binary, threshold = apply_threshold(img)

start = time.time()
scale = get_soma_scale(img, fast=True)




img_float = sitk.Cast(img, sitk.sitkFloat32)
frangi_filter = sitk.ObjectnessMeasureImageFilter()
# frangi_filter.SetScaleObjectnessMeasure(...)
frangi_filter.SetObjectDimension(1)
frangi = frangi_filter.Execute(img_float)
frangi = sitk.Cast(frangi, sitk.sitkUInt16)

seg = sitk.ConfidenceConnected(img, seedList=soma_seeds,
                               numberOfIterations=5,
                               multiplier=3,
                               initialNeighborhoodRadius=1,
                               replaceValue=1)

label_seg = sitk.ConnectedComponent(seg)

ax = fig.add_subplot(1, 2, 1)
plt.title('Frangi Vesselness')
plt.imshow(flatten(frangi), cmap='gray')
plt.colorbar()

ax = fig.add_subplot(1, 2, 2)
plt.title('Labels')
plt.imshow(flatten(label_seg), cmap='nipy_spectral', interpolation="nearest")
plt.colorbar()

plt.tight_layout()
plt.show()

sitk.WriteImage(seg, 'data/test_segmentation_binary/labels.tif')

pass

# dilate_filter = sitk.BinaryDilateImageFilter()
# dilate_filter.SetKernelRadius(radius)
# dilate_filter.SetForegroundValue(1)
# dilated = dilate_filter.Execute(binary)
#
#
# imshow_flatten(axs[0, 2], dilated, cmap='gray')
# axs[0, 2].set_title(f'Dilated R={radius}')
# axs[0, 2].set_axis_off()
#
#
#
#
# # https://stackoverflow.com/questions/40720176/how-to-extract-labels-from-a-binary-image-in-simpleitk-in-python
# label_image = sitk.ConnectedComponent(dilated)
# stats = sitk.LabelIntensityStatisticsImageFilter()
# stats.Execute(label_image, img)
#
# start_time = time.time()
# components = np.zeros(int(len(stats.GetLabels())), dtype=[('labels', np.int32), ('sizes', np.float32)])
# components['labels'] = stats.GetLabels()
# for ii, ll in enumerate(stats.GetLabels()):
#     components['sizes'][ii] = stats.GetPhysicalSize(ll)
#     # print("Label: {0} -> Mean: {1} Size: {2}".format(ll, stats.GetMean(ll), size))
# print(f'Extracting label data took: {time.time() - start_time} s')
#
# axs[1, 0].hist(components['sizes'], log=True)
# axs[1, 0].set_title('Histogram of Component Sizes')
# axs[1, 0].grid(True)
#
# # Create a set of inset Axes: these should fill the bounding box allocated to
# # them.
# axs2 = plt.axes([0, 0, 1, 1])
# # Manually set the position and relative size of the inset axes within ax1
# ip = InsetPosition(axs[1, 0], [0.4, 0.2, 0.5, 0.5])
# axs2.set_axes_locator(ip)
# # Mark the region corresponding to the inset axes on ax1 and draw lines
# # in grey linking the two axes.
# # mark_inset(axs[1, 0], axs2, loc1=2, loc2=4, fc="none", ec='0.5')
#
# size_threshold = threshold_otsu(components['sizes'])
# axs[1, 0].axvline(x=size_threshold, color='r', ls=':', label='Cut-off (Otsu)')
# neuron_components = components[components['sizes'] > size_threshold]
# axs2.hist(neuron_components['sizes'], log=False)
# axs[1, 0].legend()
#
# relabelMap = {ii: 0 for ii in stats.GetLabels() if stats.GetPhysicalSize(ii) < size_threshold}
# label_image = sitk.ChangeLabel(label_image, changeMap=relabelMap)
#
# relabel_cons_filter = sitk.RelabelComponentImageFilter()
# label_image = relabel_cons_filter.Execute(label_image)
#
# labelmap_filter = sitk.LabelImageToLabelMapFilter()
# label_map = labelmap_filter.Execute(label_image)
#
# imshow_flatten(axs[1, 1], label_image, cmap='nipy_spectral')
# axs[1, 1].set_title('Segmented Image')
# axs[1, 1].set_axis_off()
#
# axs[1, 2].imshow(volume_show(label_image, labeled=[True], pos=4))
# axs[1, 2].set_title('Segmented Image')
# axs[1, 2].set_axis_off()
#
# fig.show()

# fig.savefig(os.path.join('data', 'test_segmentation_binary', f'r{radius}.jpg'))
#
# im2 = sitk.GetArrayFromImage(label_image)
# tif.imsave(os.path.join('data', 'test_segmentation_binary', f'r{radius}.tif'), im)
