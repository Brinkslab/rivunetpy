import SimpleITK as sitk
from rivunetpy.rivunetpy import HyperStack

import tifffile

if __name__ == '__main__':

    hstack = HyperStack().from_file('data\dataset_s0_c9_4D_20dB_SNR-weird-units.tif')

    pass