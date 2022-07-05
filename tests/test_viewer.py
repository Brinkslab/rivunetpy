import SimpleITK as sitk
import os

from rivunetpy.utils.io import loadswc
from rivunetpy.swc import SWC
from rivunetpy.utils.plottools import volume_view

SWC_FOLDER = r'H:\Duet\Duet\dataset_s0_c9'
IMG = r'H:\Duet\Synthetic Data\dataset_s0_c9\MAX_dataset_s0_c9_structure.tif'


if __name__ == '__main__':

    swc_filenames = []
    for file in os.listdir(SWC_FOLDER):
        if os.path.splitext(file)[-1] == '.swc':
            swc_filenames.append(file)

    swcs = []
    for swc_fname in swc_filenames:
        swc_mat = loadswc(os.path.join(SWC_FOLDER, swc_fname))
        s = SWC()
        s._data = swc_mat

        swcs.append(s)

    img = sitk.ReadImage(IMG)

    swc_and_image = swcs
    swc_and_image.append(img)
    swc_and_image = tuple(swc_and_image)

    volume_view(*swc_and_image)