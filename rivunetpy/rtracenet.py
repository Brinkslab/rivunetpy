import os
import sys
from multiprocessing import Pool

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import SimpleITK as sitk

from rivunetpy.trace import R2Tracer
from rivunetpy.swc import SWC
from rivunetpy.utils.plottools import flatten
from rivunetpy.utils.io import loadswc, loadimg, crop
from rivunetpy.utils.filtering import apply_threshold
from rivunetpy.trace import estimate_radius
from rivunetpy.utils.segmentation import NeuronSegmentor
from rivunetpy.utils.cells import Neuron
from rivunetpy.utils.extensions import RIVULET_2_TREE_SWC_EXT, RIVULET_2_TREE_IMG_EXT


def check_long_ext(file_to_check, ext):
    return file_to_check.split(os.extsep, 1)[-1] == ext.split(os.extsep, 1)[-1]

def trace_single(neuron, threshold, speed, quality, force_retrace, scale):
    starttime = time.time()
    img = neuron.img
    neuron.swc_fname = '{}{}'.format(neuron.img_fname.split(RIVULET_2_TREE_IMG_EXT)[0], RIVULET_2_TREE_SWC_EXT)

    if os.path.exists(neuron.swc_fname) and not force_retrace:
        swc_mat = loadswc(neuron.swc_fname)
        swc = SWC()
        swc._data = swc_mat
        neuron.add_SWC(swc)
        print(f'Neuron ({neuron.num})\t --Loaded SWC from disk')
    else:
        if threshold in (float, int):
            pass
        elif type(threshold) is str:
            _, reg_thresh = apply_threshold(img, mthd=threshold)
        else:
            _, reg_thresh = apply_threshold(img, mthd='Max Entropy')

        img: np.ndarray = sitk.GetArrayFromImage(img)
        img = np.moveaxis(img, 0, -1)
        img = np.swapaxes(img, 0, 1)

        img, crop_region = crop(img, reg_thresh)  # Crop by default
        print(f'Neuron ({neuron.num})\t --Tracing neuron of shape {img.shape} '
              f'with a threshold of {reg_thresh}')

        # Run rivulet2 for the first time
        skeletonize = False
        tracer = R2Tracer(quality=quality,
                          silent=True,
                          speed=speed,
                          clean=True,
                          non_stop=False,
                          skeletonize=skeletonize)

        swc, soma = tracer.trace(img, reg_thresh)
        print('Neuron ({})\t -- Finished: {:.2f} sec.'.format(neuron.num, time.time() - starttime))

        # if skeletonized, re-estimate the radius for each node
        if skeletonize:
            print('Re-estimating radius...')
            swc_arr = swc.get_array()
            for i in range(swc_arr.shape[0]):
                swc_arr[i, 5] = estimate_radius(swc_arr[i, 2:5], img > reg_thresh)
            swc._data = swc_arr

        swc.clean()

        swc.apply_soma_TypeID(soma)

        swc.apply_scale(scale)

        swc.reset(crop_region, 1)

        swc.save(neuron.swc_fname)
        neuron.add_SWC(swc)

    return neuron

def plot(results):
    fig, ax = plt.subplots(1, 2)

    cmaps = ['hot', 'bone', 'copper', 'pink']
    num_cmaps = len(cmaps)

    neuron_fnames = [neuron.img_fname for neuron in results]

    for ii, fname in enumerate(neuron_fnames):
        img = sitk.ReadImage(fname)
        cmap = matplotlib.cm.get_cmap(cmaps[ii % num_cmaps])

        shape = img.GetSize()

        if img.GetPixelID() == sitk.sitkUInt16:
            max_px = 2**16-1
        else:
            max_px = 2**8-1

        # Create color map with step in alpha channel
        # Convert threshold to value between 0-1
        step = 1/max_px
        aa_cmap = cmap(np.arange(cmap.N))
        aa_cmap[:, -1] = np.linspace(0, 1, cmap.N) > step
        aa_cmap = ListedColormap(aa_cmap)

        ax[0].imshow(flatten(img),
                   cmap=aa_cmap,
                   extent=[0, shape[0], 0, shape[1]],
                   alpha=1)

    for neuron in results:
        swc = neuron.swc
        swc.as_image(ax=ax[1])


    fig.show()

def trace_net(file=None, dir_out=None, threshold=None, force_recalculate=False,
              speed=False, quality=False, asynchronous=True, voxel_size=()):

    if threshold is not None and type(threshold) not in (int, float, str):
        raise TypeError('Expected threshold to be either of type str, specifiying an automatic thresholding method \n',
                        f'or a number, specifying the threshold value, instead got {type(threshold)}')

    img_dir, img_name = os.path.split(file)

    if dir_out is None:
        dir_out = os.path.splitext(img_name)[0]

    save_dir = os.path.join(img_dir, dir_out)

    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file)
    file_reader.ReadImageInformation()

    img_info = [pair.split('=') for pair in file_reader.GetMetaData('ImageDescription').split('\n')]
    img_info.remove([''])
    img_info = dict(img_info)

    voxelsize = img_info.get('spacing')
    period = img_info.get('finterval')
    period_unit = img_info.get('tunit')

    z_depth = img_info.get('slices')
    frames = img_info.get('frames')

    if voxelsize is not None:
        voxelsize = float(voxelsize)

    if period is not None:
        if period_unit is not None:
            if period_unit == 'ms':
                period = float(period)
            if period_unit in ['seconds', 'sec', 's']:
                period = float(period) / 1000

    if z_depth is not None:
        z_depth = int(z_depth)

    if frames is not None:
        frames = int(frames)

    if (os.path.exists(save_dir)
            and not force_recalculate
            and any([check_long_ext(fname, RIVULET_2_TREE_IMG_EXT) for fname in os.listdir(save_dir)])):

        neurons = []
        for fname in os.listdir(save_dir):
            if check_long_ext(fname, RIVULET_2_TREE_IMG_EXT):
                loc = os.path.join(save_dir, fname)
                image = loadimg(loc, 1)
                neurons.append(Neuron(image, img_fname=loc, num=len(neurons)))
        print(f'Loaded {len(neurons)} neuron images from file.')
    else:
        ################ LOAD IMAGE AND METADATA #################
        img = loadimg(file, 1)
        X_size, Y_size, stacks = img.GetSize()

        ######## CREATE PROJECTIONS FOR TRACING AND VOLTAGE IMAGE DATA ANALYSIS ##########

        # Project to 3D image to get geometry
        T_project = np.reshape(sitk.GetArrayFromImage(img), (frames, z_depth, X_size, Y_size))
        T_project = np.amax(T_project, axis=0)


        del(img) # Get huge image out of memory ASAP

        # Project to 2D image to attain (relative) votlage data when XY coords of somata are known
        ZT_project = np.amax(T_project, axis=0)

        T_project = sitk.GetImageFromArray(T_project)
        ZT_project = sitk.GetImageFromArray(ZT_project)

        # Write image to test segmentation
        # sitk.WriteImage(T_project, 'Projected.tif')

        # Create a new directory next to the input file for the SWC outputs
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        neuronsegmentor = NeuronSegmentor(T_project)
        neurons = neuronsegmentor.neurons

        # DEBUG PLOTS
        # neuronsegmentor.plot()
        # neuronsegmentor.plot_full_segmentation()

        for ii, neuron in enumerate(neurons):
            # Store images
            img_fname = 'neuron_{:04d}{}'.format(ii, RIVULET_2_TREE_IMG_EXT)
            loc = os.path.join(save_dir, img_fname)
            neuron.img_fname = loc
            sitk.WriteImage(neuron.img, neuron.img_fname)
        print(f'Segmented image into {len(neurons)} neurons.')

    if asynchronous:
        with Pool(processes=os.cpu_count() - 1) as pool:
            result_buffers = []
            for neuron in neurons:
                result = pool.apply_async(trace_single,
                                                 (neuron, threshold, speed, quality, force_recalculate, voxelsize))
                result_buffers.append(result)

            results = [result.get() for result in result_buffers]
    else:
        results = []
        for neuron in neurons:
            results.append(trace_single(neuron, threshold, speed, quality, force_recalculate, voxelsize))

    plot(results)

    return results

