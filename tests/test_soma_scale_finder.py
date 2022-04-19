import os
import time

import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np

from rivuletpy.utils.io import loadtiff3d
from rivuletpy.utils.filtering import apply_threshold, rolling_ball_removal
from rivuletpy.utils.plottools import flatten
from rivuletpy.utils.segmentation import get_soma_scale


def resample_image(itk_image, out_spacing=1.0, is_label=False):
    # https://www.programcreek.com/python/example/96383/SimpleITK.sitkNearestNeighbor
    out_spacing = np.full(3, out_spacing)
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

plt.style.use('dark_background')



filename = 'data/Series017.v3dpbd.tif'

scales = np.round(np.arange(0.1, 3, 0.1), 2)
sizes = []
times = []

master_im = loadtiff3d(filename, out='SITK')


fig, ax = plt.subplots(1, len(scales), figsize=(10, 2.5))

for ii, scale in enumerate(scales):
    spacing = 1/scale

    im = resample_image(master_im, out_spacing=spacing)
    start = time.time()
    print(f'({ii+1})\t'
          f'Finished rescaling image from {master_im.GetSize()} to'
          f'{im.GetSize()}.\n\tProcessing time {round(time.time() - start, 1)} s')
    ax[ii].imshow(flatten(im), cmap='gray')
    ax[ii].set_title(f'{scale}X')
    ax[ii].axis('off')

    binary, _ = apply_threshold(im)

    start = time.time()
    sizes.append(get_soma_scale(im, fast=True))
    times.append(time.time() - start)
    print(f'\tFinished scale {scale}X, processing time {round(times[-1], 1)}\n')

fig.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

lin_fit = np.polyfit(scales, times, deg=1)
quad_fit = np.polyfit(np.power(scales, 2), times, deg=1)
cub_fit = np.polyfit(np.power(scales, 3), times, deg=1)

xx = scales
ax[0].plot(xx, times, label='Algorithm')
ax[0].plot(xx, lin_fit[0]*xx, label='$O(n)$')
ax[0].plot(xx, quad_fit[0]*xx**2, label='$O(n^2)$')
ax[0].plot(xx, cub_fit[0]
           *xx*xx**3, label='$O(n^3)$')
# ax[0].plot(scales, scales*np.log(scales), label='$O(n\cdot\log{(n)})$')
ax[0].set_xlabel('Scale')
ax[0].set_ylabel('Time [s]')
ax[0].legend()

ax[1].plot(scales, sizes)
ax[1].set_xlabel('Scale')
ax[1].set_ylabel('Estimated size')

fig.show()