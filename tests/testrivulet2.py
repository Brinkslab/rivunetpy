import os

from SimpleITK import ReadImage
import matplotlib.pyplot as plt
from skimage.io import imread

from rivuletpy import rtrace
from rivuletpy.swc import SWC
from rivuletpy.utils.io import loadswc
from rivuletpy.utils.plottools import flatten
from rivuletpy.utils.plottools import volume_show, volume_view

FILENAME = 'data/Image5.v3dpbd.tif'
FORCE = True  # Force recalculation of SWC

if __name__ == '__main__':

    # image = ReadImage(FILENAME, imageIO='TIFFImageIO')
    #
    # fig, ax = plt.subplots(1, 3)

    rtrace.main(file=FILENAME)

    # swc_mat = loadswc(out_name)
    # s = SWC()
    # s._data = swc_mat
    #
    # # s.as_image(ax=ax[2])
    # # ax[2].set_title('SWC')
    #
    # volume_view(s, image, swc_Z_offset=10)
    #
    # fig.show()
