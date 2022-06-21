import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

from rivunetpy.rivunetpy import HyperStack
from rivunetpy.rivunetpy import Tracer
from rivuletpy.utils.plottools import flatten


def _get_voltage_single(neuron: str, hyperstack: HyperStack, radius):
    soma_centroid = (405, 304, 8)  # XYZ of soma in cleaned SWC
    x_size, y_size, z_size, frames = hyperstack.GetSize()

    mask = sitk.Image((x_size, y_size, z_size), hyperstack.GetPixelID())

    idx = mask.TransformPhysicalPointToIndex(soma_centroid)
    mask[idx] = 1

    bin_dil_filt = sitk.BinaryDilateImageFilter()
    bin_dil_filt.SetKernelRadius(radius)
    bin_dil_filt.SetKernelType(sitk.sitkBall)
    mask = bin_dil_filt.Execute(mask)

    # plt.imshow(flatten(mask))
    # plt.show()

    intensities = np.zeros(frames)

    for ii in range(frames):
        volume = hyperstack[:, :, :, ii]
        volume = volume * mask
        intensities[ii] = np.sum(sitk.GetArrayFromImage(volume))

    return intensities

if __name__ == '__main__':

    hstack = HyperStack().from_file('data\dataset_s0_c9_4D_20dB_SNR_small.tif')

    intensities = _get_voltage_single(None, hstack, 5)

    plt.plot(intensities)
    plt.show()
    pass
