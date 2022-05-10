import os
import time

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

from rivuletpy.trace import R2Tracer
from rivuletpy.utils.io import loadtiff3d
from rivuletpy.swc import SWC
from rivuletpy.utils.io import loadswc
from rivuletpy.utils.plottools import volume_show, volume_view, flatten
from rivuletpy.utils.filtering import apply_threshold, rolling_ball_removal
from rivuletpy.utils.segmentation import NeuronSegmentor

FORCE = False

if __name__ == '__main__':

    plt.style.use('dark_background')

    filename = 'data/synthetic-3-cells.tif'

    img = loadtiff3d(filename, out='SITK') # Original Image

    swcs = []

    filenames_out = os.path.splitext(filename)[0] + str(0).zfill(3) + '.nr.tif'
    if (not os.path.exists(filenames_out)) or FORCE:
        neurons = NeuronSegmentor(img)

        for ii, neuron_img in enumerate(neurons.neurons):
            filenames_out = os.path.splitext(filename)[0] + str(ii).zfill(3) + '.nr.tif'
            sitk.WriteImage(neuron_img, filenames_out)
            print(f'Saved neuron to: {filenames_out}')

        neuron_images = neurons.neurons
    else:
        neuron_images = []
        dirname, fname = os.path.split(filename)
        fname, ext = os.path.splitext(fname) # Remove ext
        for ff in os.listdir(dirname):
            if fname in ff and (os.path.splitext(os.path.splitext(ff)[0])[-1] + os.path.splitext(ff)[-1]) == '.nr.tif':
                neuron_images.append(sitk.ReadImage(os.path.join(dirname, ff)))

    for ii, neuron_img in enumerate(neuron_images):
        out = os.path.splitext(filename)[0] + str(ii).zfill(3) + '.r2.swc'

        if (not os.path.exists(out)) or FORCE:
            print('Create new')
            threshold_filter = sitk.MaximumEntropyThresholdImageFilter()
            threshold_filter.Execute(neuron_img)

            threshold = threshold_filter.GetThreshold()

            tracer = R2Tracer(quality=False,
                              silent=False,
                              speed=False,
                              clean=True,
                              non_stop=False,
                              skeletonize=False)

            swc, soma = tracer.trace(sitk.GetArrayFromImage(neuron_img), threshold)

            swc.save(out)

        else:
            print('On disk')
            swc = SWC()
            swc._data = loadswc(out)

        print('.')
        swcs.append(swc)

    print(swcs)

    for swc in swcs:
        swc.set_fanciness(False)
        swc.set_view_density(75)

    volume_view(img, *swcs, labeled=False)

    pass


