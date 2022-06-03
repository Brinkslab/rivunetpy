import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.feature import blob_doh

from rivunetpy.utils.io import loadtiff3d
from rivunetpy.utils.plottools import flatten



if __name__ == '__main__':
    scale = 15

    filename = 'data/AVG_dataset_s0_c9-SNR_25-3.tif'

    img = loadtiff3d(filename, out='SITK')  # Original Image

    frangi_filter = sitk.ObjectnessMeasureImageFilter()
    # frangi_filter.SetGamma(1)
    # frangi_filter.SetAlpha(0.5)
    # frangi_filter.SetBeta(0.5)
    frangi_filter.SetBrightObject(True)
    frangi_filter.SetScaleObjectnessMeasure(True)
    frangi_filter.SetObjectDimension(0)

    gaussian_filter = sitk.DiscreteGaussianImageFilter()
    gaussian_filter.SetMaximumKernelWidth(1000)
    gaussian_filter.SetUseImageSpacing(False)
    gaussian_filter.SetVariance(int(scale ** 2))  # Sigma = Var^2

    img_blurred = gaussian_filter.Execute(img)
    img_blurred = sitk.RescaleIntensity(img_blurred, 0, 65535)
    img_blurred = sitk.Cast(img_blurred, sitk.sitkFloat32)

    img_blurred[[0, -1], :] = 0
    img_blurred[[0, -1], :] = 0

    # plt.imshow(flatten(img_blurred))
    # plt.title('Blurred')
    # plt.show()


    result = frangi_filter.Execute(img_blurred)

    plt.imshow(flatten(result))
    plt.title(f'Frangi scale={scale}')
    plt.show()
