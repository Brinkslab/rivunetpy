import copy

import numpy as np
from skimage.filters import threshold_otsu
from skimage import data, restoration, util
import SimpleITK as sitk

THRESHOLD_OPTIONS = {'Otsu': sitk.OtsuThresholdImageFilter,
                     'Max Entropy': sitk.MaximumEntropyThresholdImageFilter,
                     'Huang': sitk.HuangThresholdImageFilter,
                     'Intermodes': sitk.IntermodesThresholdImageFilter,
                     'IsoData': sitk.IsoDataThresholdImageFilter,
                     'Kittler-Illingworth': sitk.KittlerIllingworthThresholdImageFilter,
                     'Li': sitk.LiThresholdImageFilter,
                     'Moments': sitk.MomentsThresholdImageFilter,
                     'Reyni Entropy': sitk.RenyiEntropyThresholdImageFilter,
                     'Shanbhag': sitk.ShanbhagThresholdImageFilter,
                     'Triangle': sitk.TriangleThresholdImageFilter}


def rolling_ball_removal(img):
    img = copy.copy(img)
    img = sitk.GetArrayFromImage(img)

    background = np.zeros_like(img)

    if img.ndim == 3:  # Z stack
        # Assume z-axis is last axis
        for ii in range(img.shape[-1]):
            background[:, :, ii] = restoration.rolling_ball(img[:, :, ii])

    elif img.ndim == 2:
        background = restoration.rolling_ball(img)

    else:
        raise TypeError('Expected an image with either 2 or 3 dimensions (Z Stack) '
                        f'but got an image of {img.ndim} dimensions of shape {img.shape}')

    filtered_img = img - background
    filtered_img = sitk.GetImageFromArray(img, isVector=False)
    return filtered_img


def apply_threshold(img, mthd='Otsu'):
    if mthd not in THRESHOLD_OPTIONS:
        raise ValueError(f'Invalid method keyword: {mthd}, please use one of the following options: \n'
                         f'{list(THRESHOLD_OPTIONS.keys())}')
    if type(img) is np.ndarray:
        img = sitk.GetImageFromArray(img)

    filter = THRESHOLD_OPTIONS[mthd]()
    filter.SetInsideValue(0)
    filter.SetOutsideValue(1)
    binary = filter.Execute(img)
    threshold = filter.GetThreshold()

    return binary, int(threshold)
