import os

import numpy as np
import matplotlib.pyplot as plt
from rivunetpy.utils.io import loadswc
from rivunetpy.swc import SWC
from skimage.io import imread
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu
from skimage import data, restoration, util

import rtrace

FILENAME = 'data/Series017.v3dpbd.tif'
FORCE = True  # Force recalculation of SWC

if __name__ == '__main__':

    out_name = FILENAME.replace('.tif', '.r2.swc')
    image = imread(FILENAME)

    # if type(image.dtype) is type(np.uint8):
    #     pass
    # elif type(image.dtype) is type(np.uint16):
    #     image = image//16
    # else:
    #     raise TypeError('Expected an image with type np.uint8 or np.uint16 (8/16-bit depth), '
    #                     f'instead got image of type: {image.dtype}')

    x = image.dtype
    y = image.max()


    if image.ndim == 3:  # Z stack
        flat_image = np.max(image, axis=0)
        print(f'(A)\tFlattened image of dimension {image.shape} into {flat_image.shape}')
    elif image.ndim == 2:
        flat_image = image
        print(f'(A)\tImage size: {image.shape}')
    else:
        raise TypeError('Expected an image with either 2 or 3 dimensions (Z Stack) '
                        f'but got an image of {image.ndim} dimensions of shape {image.shape}')

    otsu = threshold_otsu(flat_image)

    for ii in range(1):

        fig, ax = plt.subplots(1, 3)

        threshold = threshold_otsu(flat_image)
        print(f'(B)\tMaximum background value (threshold) set to {threshold}')

        ax[0].imshow(flat_image, cmap='gray')
        ax[0].set_title('Input')
        ax[1].imshow(flat_image > threshold, cmap='gray')
        ax[1].set_title(f'Threshold={threshold}')

        fig.show()

