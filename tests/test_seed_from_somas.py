import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu
from skimage import data, restoration, util

from tests import rtrace
from rivuletpy.utils.io import loadswc, loadimg
from rivuletpy.swc import SWC

FILENAME = 'Synthetic-no-bg.tif'
READ_FOLDER = 'data'
WRITE_FOLDER = os.join('data', 'test_seed_from_somas')

FORCE = True  # Force recalculation of SWC


if __name__ == '__main__':

    file = os.path.join(WRITE_FOLDER, FILENAME)
    out_name = file.replace('.tif', '.r2.swc')

    image = loadimg()

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

