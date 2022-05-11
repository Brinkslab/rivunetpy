import os
import time

import numpy as np
import tifffile as tif
from scipy import stats
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from mpl_toolkits.axes_grid1.inset_locator import (InsetPosition)
import SimpleITK as sitk

from rivunetpy.utils.io import loadtiff3d
from rivunetpy.utils.plottools import volume_show, volume_view, imshow_flatten

plt.style.use('dark_background')

FONTSIZE = 5
CAMERA = {'pos': 2, 'foc': 1, 'az': 30, 'el': 45}
WINDOW = {'w': 600, 'h': 1200}

A_PRIORI_NUM = 3

num_found = []
MSE = []

filename = 'data/synthetic-3-cells.tif'


fig, axs = plt.subplots(2, 4, dpi=300)

img = sitk.ReadImage(filename)  # SimpleITK object


# http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/32_Watersheds_Segmentation.html
feature_img = sitk.GradientMagnitudeRecursiveGaussian(img, sigma=1.5)

ws_img = sitk.MorphologicalWatershed(feature_img, level=4, markWatershedLine=False, fullyConnected=False)

# Find mode of Watershed image for determining background value
ws_intensities = sitk.GetArrayFromImage(ws_img).flatten()
x = ws_img[0, 0, 0]
ws_background_val = stats.mode(ws_intensities)[0][0] # Also returns counts, just need mode

seg = sitk.ConnectedComponent(ws_img != ws_background_val)

filled = sitk.BinaryFillhole(seg!=0)
d = sitk.SignedMaurerDistanceMap(filled, insideIsPositive=False, squaredDistance=False, useImageSpacing=False)

ws = sitk.MorphologicalWatershed(d, markWatershedLine=False, level=1)
# ws = sitk.Mask( ws, sitk.Cast(seg, ws.GetPixelID()))

imshow_flatten(axs[0, 0], img, cmap='gray')
axs[0, 0].set_title('Original Image', fontsize=FONTSIZE)
axs[0, 0].set_axis_off()

imshow_flatten(axs[0, 1], feature_img, cmap='gray')
axs[0, 1].set_title('Feature Image', fontsize=FONTSIZE)
axs[0, 1].set_axis_off()

imshow_flatten(axs[0, 2], ws_img, cmap='gist_ncar')
axs[0, 2].set_title('Watershed over segmentation', fontsize=FONTSIZE)
axs[0, 2].set_axis_off()

imshow_flatten(axs[0, 3], seg, cmap='nipy_spectral')
axs[0, 3].set_title('Foreground components', fontsize=FONTSIZE)
axs[0, 3].set_axis_off()

imshow_flatten(axs[1, 0], d, cmap='gray')
axs[1, 0].set_title('Distance Map', fontsize=FONTSIZE)
axs[1, 0].set_axis_off()

imshow_flatten(axs[1, 1], ws, cmap='nipy_spectral')
axs[1, 1].set_title('Split Mask', fontsize=FONTSIZE)
axs[1, 1].set_axis_off()




fig.show()




volume_view(ws, labeled=True)

# https://stackoverflow.com/questions/40720176/how-to-extract-labels-from-a-binary-image-in-simpleitk-in-python
label_image = sitk.ConnectedComponent(binary)
stats = sitk.LabelIntensityStatisticsImageFilter()
stats.Execute(label_image, img)

start_time = time.time()
components = np.zeros(int(len(stats.GetLabels())), dtype=[('labels', np.int32), ('sizes', np.float32)])
components['labels'] = stats.GetLabels()
for ii, ll in enumerate(stats.GetLabels()):
    components['sizes'][ii] = stats.GetPhysicalSize(ll)
    # print("Label: {0} -> Mean: {1} Size: {2}".format(ll, stats.GetMean(ll), size))
print(f'Extracting label data took: {time.time() - start_time} s')

axs[1, 0].hist(components['sizes'], log=True)
axs[1, 0].set_title('Histogram of Component Sizes')
axs[1, 0].grid(True)

# Create a set of inset Axes: these should fill the bounding box allocated to
# them.
axs2 = plt.axes([0,0,1,1])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(axs[1, 0], [0.4,0.2,0.5,0.5])
axs2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
# mark_inset(axs[1, 0], axs2, loc1=2, loc2=4, fc="none", ec='0.5')

size_threshold = threshold_otsu(components['sizes'])
axs[1, 0].axvline(x=size_threshold, color='r', ls=':', label='Cut-off (Otsu)')
neuron_components = components[components['sizes'] > size_threshold]
axs2.hist(neuron_components['sizes'], log=False)
axs[1, 0].legend()


relabelMap = {ii : 0 for ii in stats.GetLabels() if stats.GetPhysicalSize(ii) < size_threshold }
label_image = sitk.ChangeLabel(label_image, changeMap=relabelMap)

relabel_cons_filter = sitk.RelabelComponentImageFilter()
label_image = relabel_cons_filter.Execute(label_image)



labelmap_filter = sitk.LabelImageToLabelMapFilter()
label_map = labelmap_filter.Execute(label_image)

# sitk.WriteImage(output, 'labels.tif')

# dilate_filter = sitk.LabelSetDilateImageFilter()
# dilate_filter.SetKernelRadius(3)
# dilate_filter.SetKernelType(sitk.sitkBall)
#
# dilate_filter.Execute(cc)

axs[1, 1].imshow(volume_show(label_image, labeled=True, **WINDOW, **CAMERA))
axs[1, 1].set_title('Segmented Image')
axs[1, 1].set_axis_off()

# volume_view(label_image, labeled=True)

fig.show()
fig.savefig(os.path.join('data', 'test_segmentation_binary', f'{threshold_method}.jpg'))

im2 = sitk.GetArrayFromImage(label_image)
tif.imsave(os.path.join('data', 'test_segmentation_binary', f'{threshold_method}.tif'), im)

unique_labels = np.unique(sitk.GetArrayFromImage(label_image))
num_found.append(len(unique_labels) - 1)

mask = np.flip(np.swapaxes(im2.astype(bool),0,2), axis=1)
im3 = mask * im

# fig, ax = plt.subplots(1, 3)
# ax[0].imshow(np.max(mask, axis=2))
# ax[1].imshow(np.max(im, axis=2))
# ax[2].imshow(np.max(im3, axis=2))
#
# fig.show()

MSE.append(np.mean(np.square(im3 - im)))


