import os
import sys
from multiprocessing import Pool
import warnings

import time
import numpy as np
import matplotlib.pyplot as plt

import SimpleITK as sitk
from SimpleITK.SimpleITK import Image

from rivunetpy.trace import R2Tracer
from rivunetpy.swc import SWC
from rivunetpy.utils.plottools import flatten, plot_swcs, plot_segmentation
from rivunetpy.utils.io import loadswc, loadimg, crop
from rivunetpy.utils.filtering import apply_threshold
from rivunetpy.trace import estimate_radius
from rivunetpy.utils.segmentation import NeuronSegmentor
from rivunetpy.utils.cells import Neuron
from rivunetpy.utils.extensions import RIVULET_2_TREE_SWC_EXT, RIVULET_2_TREE_IMG_EXT


def check_long_ext(file_to_check, ext):
    return file_to_check.split(os.extsep, 1)[-1] == ext.split(os.extsep, 1)[-1]


class VITracer():
    # TODO: Clean up new class-based mehtods
    def __init__(self):
        self.filename = None
        self.out = None
        self.threshold = None
        self.force_redo = False
        self.quality = False
        self.asynchronous = True
        self.neurons = None

    def set_file(self, fname: str):
        self.fname = fname

        img_dir, img_name = os.path.split(self.fname)

        if self.out is None:
            self.out = os.path.splitext(img_name)[0]
            self.out = os.path.join(img_dir, self.out)

    def set_output_dir(self, out: str):
        self.out = out

    def set_threshold(self, threshold: int):
        self.threshold = threshold

    def force_redo_on(self):
        self.force_redo = True

    def force_redo_off(self):
        self.force_redo = False

    def set_force_redo(self, force: bool):
        self.force_redo = force

    def quality_on(self):
        self.quality = True

    def quality_off(self):
        self.quality = False

    def set_quality(self, quality: bool):
        self.quality = quality

    def asynchronous_on(self):
        self.asynchronous = True

    def asynchronous_off(self):
        self.asynchronous = False

    def set_asynchronous(self, asynchronous: bool):
        self.asynchronous = asynchronous

    def plot(self, results, units=''):
        fig, ax = plt.subplots(1, 2)

        plot_segmentation(results, ax=ax[0])

        ax[0].set_xlabel('X [px]')
        ax[0].set_ylabel('Y [px]')

        swcs = []
        for neuron in results:
            swcs.append(neuron.swc)

        plot_swcs(swcs, ax=ax[1], units=units)

        fig.show()

    @staticmethod
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

            swc.reset(crop_region, 1)

            swc.apply_scale(scale)

            swc.save(neuron.swc_fname)
            neuron.add_SWC(swc)

        return neuron

    @staticmethod
    def get_voltage_single(neuron: Neuron, img: Image, radius):
        soma_centroid = neuron.swc._data[0, 2:5]  # XYZ of soma in SWC

        mask = sitk.Image(img.GetSize(), img.GetPixelID())

        idx = mask.TransformPhysicalPointToIndex(soma_centroid)
        mask[idx] = 1

        bin_dil_filt = sitk.BinaryDilateImageFilter()
        bin_dil_filt.SetKernelRadius(radius)
        bin_dil_filt.SetKernelType(sitk.sitkBall)
        mask = bin_dil_filt.Execute(mask)

        plt.imshow(flatten(mask))
        plt.show()

    def get_voltage(self):
        img = loadimg(self.file, 1)

        if self.asynchronous:
            with Pool(processes=os.cpu_count() - 1) as pool:
                result_buffers = []
                for neuron in self.neurons:
                    result = pool.apply_async(self.get_voltage_single,
                                              (neuron, img, 20))
                    result_buffers.append(result)

                results = [result.get() for result in result_buffers]
        else:
            results = []
            for neuron in self.neurons:
                results.append(self.get_voltage_single(neuron, img, 20))

    def _must_read_segmentation_file(self):
        (os.path.exists(save_dir)
         and not self.force_recalculate
         and any([check_long_ext(fname, RIVULET_2_TREE_IMG_EXT) for fname in os.listdir(save_dir)]))

    def _read_segmentation_from_file(self):


    def execute(self):

        if self.threshold is not None and not isinstance(self.threshold, (int, float, str)):
            raise TypeError(
                'Expected threshold to be either of type str, specifiying an automatic thresholding method \n',
                f'or a number, specifying the threshold value, instead got {type(self.threshold)}')





        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(self.file)
        file_reader.ReadImageInformation()

        if 'ImageDescription' in file_reader.GetMetaDataKeys():
            img_info = [pair.split('=') for pair in file_reader.GetMetaData('ImageDescription').split('\n')]
            img_info.remove([''])
            img_info = dict(img_info)
        else:
            img_info = {}
            warnings.warn('rtracenet: Warning, could not read metadata from image. Consider using a tool such as \n'
                          'ImageJ to set the metadata. Ignoring this error might lead to unexpected behavior.')

        is_hyperstack = img_info.get('hyperstack')

        voxelsize = img_info.get('spacing')
        voxel_unit = img_info.get('unit')
        voxel_unit_str = None

        period = img_info.get('finterval')
        period_unit = img_info.get('tunit')
        period_unit_str = None

        z_depth = img_info.get('slices')
        frames = img_info.get('frames')

        if is_hyperstack is not None:
            if is_hyperstack == 'false':
                raise IOError(
                    'Input image should be 4D voltage imaging data. The first three dimensions should be from \n'
                    'a 3D confocal image stack. The final fourth dimension should be time. \n'
                    'Finally, please format the image as a hyperstack in ImageJ.')

        if voxelsize is not None and voxel_unit is not None:
            if voxel_unit == '\\u00B5m':  # Micrometers
                voxel_unit_str = '$\\mu\\mathrm{m}$'
                voxelsize = float(voxelsize)

            elif voxel_unit == 'mm':
                voxel_unit_str = '$\\mu\\mathrm{m}$'  # Convert to micrometers
                voxelsize = float(voxelsize * 1E3)

            elif voxel_unit == 'nm':
                voxel_unit_str = '$\\mu\\mathrm{m}$'  # Convert to micrometers
                voxelsize = float(voxelsize / 1E3)

            else:
                raise IOError(f'Input image has voxels with unknown unit {voxel_unit}. \n'
                              'Please use a tool such as ImageJ to specify the voxel size.')
        else:
            warnings.warn('rtracenet: Warning, no voxel size found, using pixel units for SWC files instead.')
            voxel_unit_str = 'px'
            voxelsize = 1

        if period is not None and period_unit is not None:
            if period_unit == 'ms':
                period_unit_str = 'ms'
                period = float(period)

            elif period_unit in ['seconds', 'sec', 's']:
                period_unit_str = 'ms'
                period = float(period) * 1000

            elif period_unit == '\\u00B5s':
                period_unit_str = 'ms'
                period = float(period) / 1000

            else:
                raise IOError(f'Input image has timescale with unknown unit {period_unit}. \n'
                              'Please use a tool such as ImageJ to specify the voxel size.')

        else:
            warnings.warn('rtracenet: Warning, no time unit found, using unitless timescales instead.')
            period_unit_str = '1'
            period = 1

        if z_depth is not None:
            z_depth = int(z_depth)

        if frames is not None:
            frames = int(frames)

        if :

            neurons = []
            for fname in os.listdir(save_dir):
                if check_long_ext(fname, RIVULET_2_TREE_IMG_EXT):
                    loc = os.path.join(save_dir, fname)
                    image = loadimg(loc, 1)
                    neurons.append(Neuron(image, img_fname=loc, num=len(neurons)))
            print(f'Loaded {len(neurons)} neuron images from file.')
        else:
            ################ LOAD IMAGE AND METADATA #################
            img = loadimg(self.file, 1)
            X_size, Y_size, stacks = img.GetSize()

            ######## CREATE PROJECTIONS FOR TRACING AND VOLTAGE IMAGE DATA ANALYSIS ##########

            # Project to 3D image to get geometry
            T_project = np.reshape(sitk.GetArrayFromImage(img), (frames, z_depth, X_size, Y_size))
            T_project = np.amax(T_project, axis=0)

            del (img)  # Get huge image out of memory ASAP

            T_project = sitk.GetImageFromArray(T_project)

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

        if self.asynchronous:
            with Pool(processes=os.cpu_count() - 1) as pool:
                result_buffers = []
                for neuron in neurons:
                    result = pool.apply_async(self.trace_single,
                                              (neuron,
                                               self.threshold,
                                               self.speed,
                                               self.quality,
                                               self.force_recalculate,
                                               self.voxelsize))
                    result_buffers.append(result)

                results = [result.get() for result in result_buffers]
        else:
            results = []
            for neuron in neurons:
                results.append(neuron,
                               self.threshold,
                               self.speed,
                               self.quality,
                               self.force_recalculate,
                               self.voxelsize))

        self.get_voltage(file, results, asynchronous=False)

        self.plot(results, units=voxel_unit_str)

        return results
