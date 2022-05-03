import time

import SimpleITK as sitk

from rivuletpy.utils.io import loadtiff3d

# ~~~~~~~~~~~~~~ RESULTS ~~~~~~~~~~~~~~~~
# No tweaks: 24.821630716323853 s
# Squared Distance: 24.44663166999817 s
# Image Spacing ON: 25.351212739944458 s
# Image Spacing OFF: 25.6224844455719 s
# GetVoronoi: 0.0 ms


if __name__ == '__main__':

    img = loadtiff3d('data/Synthetic-no-bg.tif', out='SITK')  # Original Image

    threshold_filter = sitk.MaximumEntropyThresholdImageFilter()
    threshold_filter.SetInsideValue(1)
    threshold_filter.SetOutsideValue(0)
    threshold_filter.Execute(img)

    threshold = threshold_filter.GetThreshold()

    components = sitk.ConnectedComponent(img > threshold)

    start = time.time()
    dist_filter = sitk.SignedDanielssonDistanceMapImageFilter()
    dist_filter.Execute(components)
    print(f'No tweaks: {(time.time()-start)} s')

    start = time.time()
    dist_filter = sitk.SignedDanielssonDistanceMapImageFilter()
    dist_filter.SetSquaredDistance(True)
    dist_filter.Execute(components)
    print(f'Squared Distance: {(time.time() - start)} s')

    start = time.time()
    dist_filter = sitk.SignedDanielssonDistanceMapImageFilter()
    dist_filter.SetUseImageSpacing(True)
    dist_filter.Execute(components)
    print(f'Image Spacing ON: {(time.time() - start)} s')

    start = time.time()
    dist_filter = sitk.SignedDanielssonDistanceMapImageFilter()
    dist_filter.SetUseImageSpacing(True)
    dist_filter.Execute(components)
    print(f'Image Spacing OFF: {(time.time() - start)} s')

    start = time.time()
    voronoi = dist_filter.GetVoronoiMap()
    print(f'GetVoronoi: {1E3 * (time.time() - start)} ms')
