import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from rivuletpy.utils.io import loadswc
from rivuletpy.swc import SWC
from skimage.io import imread
from skimage.filters import threshold_otsu

from rivuletpy import rtrace

FILENAME = 'data/Series017.v3dpbd.tif'
FORCE = True  # Force recalculation of SWC

if __name__ == '__main__':

    out_name = FILENAME.replace('.tif', '.r2.swc')
    image = imread(FILENAME)

    image = (image // 16).astype(np.uint8)

    print(f'(I)\tImage of type {image.dtype} with a intensity extrema (max, min) of, '
          f'{(image.min(), image.max())}')

    # if type(image.dtype) is type(np.uint8):
    #     pass
    # elif type(image.dtype) is type(np.uint16):
    #     image = image//16
    # else:
    #     raise TypeError('Expected an image with type np.uint8 or np.uint16 (8/16-bit depth), '
    #                     f'instead got image of type: {image.dtype}')

    for ii in range(75, 125, 5):

        fig, ax = plt.subplots(1, 3)

        if image.ndim == 3:  # Z stack
            flat_image = np.max(image, axis=0)
            print(f'(A)\tFlattened image of dimension {image.shape} into {flat_image.shape}')
        elif image.ndim == 2:
            flat_image = image
            print(f'(A)\tImage size: {image.shape}')
        else:
            raise TypeError('Expected an image with either 2 or 3 dimensions (Z Stack) '
                            f'but got an image of {image.ndim} dimensions of shape {image.shape}')

        threshold = threshold_otsu(flat_image)
        threshold_old = threshold
        threshold = int(threshold*ii/100)
        print(f'(B)\tMaximum background value (threshold) set to {ii}% of Otsu-deterimined '
            f' value of {threshold_old}: set to {threshold}')

        ax[0].imshow(flat_image, cmap='gray')
        ax[0].set_title('Input')
        ax[1].imshow(flat_image > threshold, cmap='gray')
        ax[1].set_title(f'Threshold={threshold}')

        if (not os.path.exists(out_name)) or FORCE:
            nn = 'data/testrivulet2-swcs/' + str(threshold).zfill(4) + '-' + out_name.replace('data/', '')
            print(f'(C)\tSaving to: {nn}')
            rtrace.main(file=FILENAME, threshold=threshold, out=out_name)

        swc_mat = loadswc(out_name)
        s = SWC()
        s._data = swc_mat
        s.as_image(ax=ax[2])
        ax[2].set_title('SWC')

        s.save(nn)

        ff = 'data/testrivulet2-images/' + str(threshold).zfill(4) + '-' + FILENAME.replace('.tif', '.png').replace('data/',
                                                                                                             '')
        print(f'(D)\tSaving plot to: {ff}')
        fig.savefig(ff)
        Image.open(ff).convert('RGB').save(ff.replace('.png', '.jpg'), 'JPEG')
        os.remove(ff)
        fig.show()

