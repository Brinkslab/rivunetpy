import os
import time

import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
import mpl_toolkits.axisartist
import SimpleITK as sitk

from rivuletpy.utils.io import loadimg, loadtiff3d
from tests.volume_rendering_vtk import volume_view, volume_show

plt.style.use('dark_background')


CAMERA = {'pos': 2, 'foc': 1, 'az': 30, 'el': 45}
WINDOW = {'w': 600, 'h': 1200}

A_PRIORI_NUM = 3

num_found = []
MSE = []

filename = 'data/Synthetic-no-bg.tif'

THRESHOLD_OPTIONS = {'Otsu' : sitk.OtsuThresholdImageFilter,
                     'Max Entropy' : sitk.MaximumEntropyThresholdImageFilter,
                     'Huang' : sitk.HuangThresholdImageFilter,
                     'Intermodes' : sitk.IntermodesThresholdImageFilter,
                     'IsoData' : sitk.IsoDataThresholdImageFilter,
                     'Kittler-Illingworth' : sitk.KittlerIllingworthThresholdImageFilter,
                     'Li' : sitk.LiThresholdImageFilter,
                     'Moments' : sitk.MomentsThresholdImageFilter,
                     'Reyni Entropy' : sitk.RenyiEntropyThresholdImageFilter,
                     'Shanbhag' : sitk.ShanbhagThresholdImageFilter,
                     'Triangle' : sitk.TriangleThresholdImageFilter}

# THRESHOLDS = THRESHOLD_OPTIONS.keys()
THRESHOLDS = list(THRESHOLD_OPTIONS.keys())

for threshold_method in THRESHOLDS:

    fig, axs = plt.subplots(2, 2, dpi=300)
    fig.suptitle(f'{threshold_method}', fontsize=12)

    threshold_func = THRESHOLD_OPTIONS[threshold_method]

    im = loadtiff3d(filename)

    img = sitk.ReadImage(filename)  # SimpleITK object

    binary_threshold = threshold_func()
    binary_threshold.SetInsideValue(0)
    binary_threshold.SetOutsideValue(1)
    binary = binary_threshold.Execute(img)

    axs[0, 0].imshow(volume_show(img, **WINDOW, pos=4))
    axs[0, 0].set_title('Original Image')
    axs[0, 0].set_axis_off()

    axs[0, 1].imshow(volume_show(binary, **WINDOW, **CAMERA))
    axs[0, 1].set_title(f'Binary Image ({threshold_method})')
    axs[0, 1].set_axis_off()

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


fig3, ax3 = plt.subplots(2, 1, sharex=True)
ax3[0].bar(THRESHOLDS, num_found)
ax3[0].axhline(y=A_PRIORI_NUM, color='r', ls=':', label='True Number')
ax3[0].set_title('Number of Recognized Cells')

ax3[1].bar(THRESHOLDS, MSE, log=True)
ax3[1].set_xticklabels(THRESHOLDS, rotation=45)
ax3[1].set_title('MSE w.r.t. Original')

fig3.show()
fig3.savefig(os.path.join('data', 'test_segmentation_binary', 'Results.jpg'))



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# start_time = time.time()
#
# seed = (132, 142, 96)
# feature_img = sitk.GradientMagnitudeRecursiveGaussian(img, sigma=.5)
# speed_img = sitk.BoundedReciprocal(feature_img)  # This is parameter free unlike the Sigmoid
#
#
#
# # axs[1, 0].imshow(volume_show(speed_img, **WINDOW, **CAMERA))
#
# fm_filter = sitk.FastMarchingBaseImageFilter()
# fm_filter.SetTrialPoints([seed])
# fm_filter.SetStoppingValue(1000)
# fm_img = fm_filter.Execute(speed_img)
#
#
#
# # axs[1, 1].imshow(volume_show(fm_img, **WINDOW, **CAMERA))
# print(f'Fast Marching done in: {time.time() - start_time} s')
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# fig.show()
#
#
#
