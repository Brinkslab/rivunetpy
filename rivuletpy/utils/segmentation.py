import copy

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from rivuletpy.utils.filtering import apply_threshold
from rivuletpy.utils.plottools import flatten


def get_intensity(img):
    if type(img) is sitk.Image:
        img = sitk.GetArrayFromImage(img)

    return np.sum(img)


def get_soma_scale(img, thresh='Otsu', fast=True):
    # This function is WAY too slow
    binary, _ = apply_threshold(img, mthd=thresh)

    if fast:
        distance_transform = sitk.SignedMaurerDistanceMapImageFilter()
        # distance_transform.SetUseImageSpacing(False)
        distance_transform.SetInsideIsPositive(True)
        # distance_transform.SetBackgroundValue(1)
        distance_transform.SetSquaredDistance(False)
        distance_img = distance_transform.Execute(binary)

        max_filter = sitk.MinimumMaximumImageFilter()
        _ = max_filter.Execute(distance_img)

        return int(max_filter.GetMaximum())
    else:
        erosions = 0

        eroded = copy.copy(binary)
        prev_eroded = eroded

        intensities = [get_intensity(eroded)]
        components = [-1] # Dummy number that should never be physically possible

        # dilate_filter = sitk.BinaryDilateImageFilter()
        # dilate_filter.SetKernelRadius(1)
        # dilate_filter.SetForegroundValue(1)

        erode_filter = sitk.BinaryErodeImageFilter()
        erode_filter.SetKernelRadius(1)
        erode_filter.SetForegroundValue(1)

        while True:
            eroded = erode_filter.Execute(eroded)

            intensities.append(get_intensity(eroded))

            label_image = sitk.ConnectedComponent(eroded)
            stats = sitk.LabelIntensityStatisticsImageFilter()
            stats.Execute(label_image, img)
            components.append(len(stats.GetLabels()))

            if components[-1] == components[-2]:
                break
            elif intensities[-1] == intensities[-2]:
                break # Back-up exit for when component finding fails

            erosions += 1

    return erosions

def get_neurite_scale(img, soma_scale):

    binary = apply_threshold(img, mthd='Otsu')
    plt.plot(flatten(binary))
    plt.show()


























