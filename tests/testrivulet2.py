import os

from SimpleITK import ReadImage
import matplotlib.pyplot as plt
from skimage.io import imread

from rivunetpy import rtrace
from rivunetpy.swc import SWC
from rivunetpy.utils.io import loadswc
from rivunetpy.utils.plottools import flatten
from rivunetpy.utils.plottools import volume_show, volume_view

FILENAME = 'data/test_swc_postprocessing/neuron_0001.r2t.tif'
FORCE = True  # Force recalculation of SWC

if __name__ == '__main__':

    rtrace.main(file=FILENAME, save_soma=True)

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
