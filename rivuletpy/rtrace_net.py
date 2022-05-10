import os
from typing import Callable

import time
from scipy.ndimage.interpolation import zoom
import SimpleITK as sitk

from rivuletpy.trace import R2Tracer
from rivuletpy.utils.io import loadimg, crop, swc2world, swc2vtk
from rivuletpy.utils.filtering import apply_threshold
from rivuletpy.trace import estimate_radius
from rivuletpy.utils.segmentation import NeuronSegmentor
from rivuletpy.utils.cells import Neuron
from rivuletpy.rtrace import show_logo

FORCE_RETRACE = False
RIVULET_2_TREE_IMG_EXT = '{}r2t{}tif'.format(os.extsep, os.extsep)
RIVULET_2_TREE_SWC_EXT = '{}r2t{}swc'.format(os.extsep, os.extsep)


def check_long_ext(file_to_check, ext):
    return file_to_check.split(os.extsep, 1)[-1] == ext.split(os.extsep, 1)[-1]

def trace_net(file=None, dir_out=None, threshold=None,
              speed=False, quality=False, clean=True, non_stop=False,
              silent=False, skeletonize=False):

    img_dir, img_name = os.path.split(file)
    img = loadimg(file, 1)

    if dir_out is None:
        dir_out = os.path.splitext(img_name)[0]

    save_dir = os.path.join(img_dir, dir_out)

    if (os.path.exists(save_dir)
            and not FORCE_RETRACE
            and any([check_long_ext(fname, RIVULET_2_TREE_IMG_EXT) for fname in os.listdir(save_dir)])):
        neurons = []
        for fname in os.listdir(save_dir):
            if check_long_ext(fname, RIVULET_2_TREE_IMG_EXT):
                loc = os.path.join(save_dir, fname)
                image = loadimg(loc, 1)
                neurons.append(Neuron(image, img_fname=loc, num=len(neurons)))
        print(f'Loaded {len(neurons)} neurons from file.')
    else:
        # Create a new directory next to the input file for the SWC outputs
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        neuronsegmentor = NeuronSegmentor(img)
        neurons = neuronsegmentor.neurons

        for ii, neuron in enumerate(neurons):
            # Store images
            img_fname = 'neuron_{:04d}{}'.format(ii, RIVULET_2_TREE_IMG_EXT)
            loc = os.path.join(save_dir, img_fname)
            neuron.img_fname = loc
            sitk.WriteImage(neuron.img, neuron.img_fname)
        print(f'Segmented image into {len(neurons)} neurons.')

    for neuron in neurons:
        starttime = time.time()
        img = neuron.img
        neuron.swc_fname = '{}{}'.format(neuron.img_fname.split(RIVULET_2_TREE_IMG_EXT)[0], RIVULET_2_TREE_SWC_EXT)

        if threshold is None:
            _, threshold = apply_threshold(img, mthd='Max Entropy')
        elif threshold in (float, int):
            pass
        elif type(threshold) is str:
            _, threshold = apply_threshold(img, mthd=threshold)
        else:
            raise TypeError('Expected threshold to be either of type str, specifiying an automatic thresholding method',
                            f'or a number, specifying the threshold value, instead got {type(threshold)}')

        print(f'Tracing neuron {neuron.num} with a threshold of {threshold}. Saving to:\n' 
              f'{neuron.swc_fname}')

        img = sitk.GetArrayFromImage(img)
        img, crop_region = crop(img, threshold)  # Crop by default

        # Run rivulet2 for the first time

        tracer = R2Tracer(quality=quality,
                          silent=silent,
                          speed=speed,
                          clean=clean,
                          non_stop=non_stop,
                          skeletonize=skeletonize)

        swc, soma = tracer.trace(img, threshold)
        print('-- Finished: %.2f sec.' % (time.time() - starttime))

        # if skeletonized, re-estimate the radius for each node
        if skeletonize:
            print('Re-estimating radius...')
            swc_arr = swc.get_array()
            for i in range(swc_arr.shape[0]):
                swc_arr[i, 5] = estimate_radius(swc_arr[i, 2:5], img > threshold)
            swc._data = swc_arr

        swc.reset(crop_region, 1)
        swc.save(neuron.swc_fname)
        print('Done')
