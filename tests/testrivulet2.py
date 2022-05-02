import os

from SimpleITK import ReadImage
import matplotlib.pyplot as plt
from skimage.io import imread

from rivuletpy import rtrace
from rivuletpy.swc import SWC
from rivuletpy.utils.io import loadswc
from rivuletpy.utils.plottools import imshow_flatten
from rivuletpy.utils.plottools import volume_show, volume_view, swc_view

FILENAME = 'data/Image5.v3dpbd.tif'
FORCE = False  # Force recalculation of SWC

if __name__ == '__main__':

    out_name = FILENAME.replace('.tif', '.swc')
    image = ReadImage(FILENAME, imageIO='TIFFImageIO')

    fig, ax = plt.subplots(1, 3)

    # imshow_flatten(ax[0], image, cmap='gray')
    # ax[0].set_title('Input')
    # imshow_flatten(ax[1], image, cmap='gray')
    # ax[1].set_title(f'Threshold={0}')

    if (not os.path.exists(out_name)) or FORCE:
        rtrace.main(file=FILENAME, threshold='Otsu', out=out_name)

    swc_mat = loadswc(out_name)
    s = SWC()
    s._data = swc_mat

    # s.as_image(ax=ax[2])
    # ax[2].set_title('SWC')

    volume_view(s, image, swc_Z_offset=10)

    fig.show()
