import os
import sys
from multiprocessing import Pool
import warnings
import time

import numpy as np
import matplotlib.pyplot as plt
import tifffile
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

from contextlib import redirect_stdout


def check_long_ext(file_to_check, ext):
    return file_to_check.split(os.extsep, 1)[-1] == ext.split(os.extsep, 1)[-1]

def convert_hyperstack_to_4D_image(img: sitk.Image, z_depth, frames):
    X_size, Y_size, stacks = img.GetSize()
    return sitk.GetImageFromArray(
        np.reshape(
            sitk.GetArrayFromImage(img), (frames, z_depth, X_size, Y_size)
        ), isVector=False)

def tifffile_read_metadata(file):
    with tifffile.TiffFile(file) as tif:
        x_resolution, factor = tif.pages[0].tags['XResolution'].value
        x_voxel_size = factor / x_resolution

        y_resolution, factor = tif.pages[0].tags['YResolution'].value
        y_voxel_size = factor / y_resolution

    return x_voxel_size, y_voxel_size


def sitk_read_metadata(img_or_file, key=None):

    if isinstance(img_or_file, str):
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(img_or_file)
        file_reader.ReadImageInformation()

        img_or_reader = file_reader

    elif isinstance(img_or_file, Image):

        img_or_reader = img_or_file

    else:
        raise ValueError('Could not read metadata from object. Input should be either a filename to read from \n' 
                         f'or an SimpleITK.Image object. Input object is of unusable type: {type(img_or_file)}.')

    metadatakeys = img_or_reader.GetMetaDataKeys()

    if key:
        keys = [key]
    else:
        keys = [key for key in metadatakeys]

    all_info = {}
    for key_to_read in keys:
        if key_to_read in metadatakeys:
            img_info = [pair.split('=') for pair in img_or_reader.GetMetaData(key_to_read).split('\n')]
            img_info.remove([''])
            img_info = dict(img_info)

            all_info.update(img_info)
        else:
            warnings.warn(f'rtracenet: Warning, could not read metadata with key {key_to_read} from image. \n'
                          f'Consider using a tool such as ImageJ to set the metadata. \n '
                          f'Ignoring this error might lead to unexpected behavior.')

    return all_info

class HyperStack(Image):

    def __init__(self, *args, **kwargs):
        self.frames = None
        self.z_size = None
        self.voxel_size = None
        self.voxel_unit_str = None
        self.period = None
        self.period_unit_str = None

        super().__init__(*args, **kwargs)


    def _add_metadata(self, metadata_dict):

        is_hyperstack = metadata_dict.get('hyperstack')

        x_voxel_size = metadata_dict['x_voxel_size'] # um/px
        y_voxel_size = metadata_dict['y_voxel_size'] # um/px

        z_voxel_size = metadata_dict.get('spacing')
        z_voxel_unit = metadata_dict.get('unit')
        voxel_unit_str = '$\\mu\\mathrm{m}$' # um, for plotting

        period = metadata_dict.get('finterval')
        period_unit = metadata_dict.get('tunit')
        period_unit_str = 'ms'

        z_size = metadata_dict.get('slices')
        frames = metadata_dict.get('frames')

        if is_hyperstack is not None:
            if is_hyperstack == 'false':
                raise IOError(
                    'Input image should be 4D voltage imaging data. The first three dimensions should be from \n'
                    'a 3D confocal image stack. The final fourth dimension should be time. \n'
                    'Finally, please format the image as a hyperstack in ImageJ.')

        if z_voxel_size is not None and z_voxel_unit is not None:
            if z_voxel_unit == '\\u00B5m':  # Micrometers
                z_voxel_size = float(z_voxel_size)

            elif z_voxel_unit == 'mm':
                z_voxel_size = float(z_voxel_size) * 1E3

            elif z_voxel_unit == 'nm':
                z_voxel_size = float(z_voxel_size) / 1E3

            else:
                raise IOError(f'Input image has voxels with unknown unit {z_voxel_size}. \n'
                              'Please use a tool such as ImageJ to specify the voxel size.')
        else:
            warnings.warn('HyperStack: Warning, no voxel size found, using pixel units for instead.')
            voxel_unit_str = 'px' # Override
            z_voxel_size = 1

            # Also convert XY voxel sizes back to pixel units
            x_voxel_size = 1
            y_voxel_size = 1

        if period is not None and period_unit is not None:
            if period_unit == 'ms':
                period = float(period)

            elif period_unit in ['seconds', 'sec', 's']:
                period = float(period) * 1000

            elif period_unit in ['\\u00B5s', 'us']:
                period = float(period) / 1000

            else:
                raise IOError(f'Input image has timescale with unknown unit {period_unit}. \n'
                              'Please use a tool such as ImageJ to specify the voxel size.')

        else:
            warnings.warn('HyperStack: Warning, no time unit found, using unitless timescales instead.')
            period_unit_str = '1'
            period = 1

        if z_size is not None:
            z_size = int(z_size)

        if frames is not None:
            frames = int(frames)

        self.frames = frames
        self.z_size = z_size
        self.voxel_size = (x_voxel_size, y_voxel_size, z_voxel_size)
        self.voxel_unit_str = voxel_unit_str
        self.period = period
        self.period_unit_str = period_unit_str
        self.x_size, self.y_size = self.GetSize()[:2]

        vx_u = 'um' if self.voxel_unit_str == '$\\mu\\mathrm{m}$' else self.voxel_unit_str

        print('Imported hyperstack with metadata:\n'
              f'\tVoxel size: {self.voxel_size[0]:.4f} {vx_u} * {self.voxel_size[1]:.4f} {vx_u} '
              f'* {self.voxel_size[2]:.4f} {vx_u}\n'
              f'\tImage size: {self.x_size} px * {self.y_size} px * {self.z_size} px \n' 
              f'\tWith {self.frames} frames.')

    @classmethod
    def from_file(cls, fname, metadata_only=False):
        # Hacky bootstrapping to read the file
        if not metadata_only:
            sitk.ProcessObject_SetGlobalWarningDisplay(False)
            img = sitk.ReadImage(fname)
            sitk.ProcessObject_SetGlobalWarningDisplay(True)

            metadata_dict = sitk_read_metadata(img, key='ImageDescription')
        else:
            metadata_dict = sitk_read_metadata(fname, key='ImageDescription')

        x_voxel_size, y_voxel_size = tifffile_read_metadata(fname)

        metadata_dict['x_voxel_size'] = x_voxel_size
        metadata_dict['y_voxel_size'] = y_voxel_size

        z_depth = metadata_dict.get('slices')
        frames = metadata_dict.get('frames')

        if z_depth is None or frames is None:
            raise IOError('Error while loading hyperstack. Metadata was found but could not by interpreted. \n' 
                          'Try editing the orignal image in ImageJ.')

        z_depth = int(z_depth)
        frames = int(frames)

        assert metadata_dict.get('hyperstack') == 'true', ('Error while loading hyperstack. Image is not flagged as \n' 
                                                      'hyperstack. Try editing it in ImageJ.')
        if metadata_only:
            hyperstack = cls()
        else:
            X_size, Y_size, stacks = img.GetSize()

            assert stacks == (frames * z_depth), \
                (f'Image dimensions do not match metadata. Number of stacks: {stacks} should equal \n'
                 'the number of Z-stacks * number of frames: \n'
                 f'({frames} * {z_depth} = {frames * z_depth} != {stacks}')

            hyperstack = cls((X_size, Y_size, z_depth, frames), img.GetPixelID())

            img = convert_hyperstack_to_4D_image(img, z_depth, frames)

            hyperstack[:, :, :, :] = img[:, :, :, :]

        hyperstack._add_metadata(metadata_dict)

        return hyperstack

    def t_project(self, mode: str = 'MAX') -> sitk.Image:
        if mode == 'MAX':
            function = np.amax
        elif mode == 'AVG':
            function = np.mean
        else:
            raise ValueError(f'Unknown projection mode: {mode}. Please use either MAX or AVG for maximum intensity \n' 
                             'and average intensity projection resp.')

        X_size, Y_size, _, _  = self.GetSize()

        # Project to 3D image to get geometry
        T_project = sitk.GetArrayFromImage(self)
        T_project = function(T_project, axis=0, keepdims=False)

        return sitk.GetImageFromArray(T_project, isVector=False)

class Tracer():
    # TODO: Clean up new class-based methods
    def __init__(self):
        self.filename = None
        self.out = None
        self.threshold = None
        self.tolerance = 0.2
        self.force_redo = False
        self.quality = False
        self.asynchronous = True
        self.neurons = None
        self._speed = False
        self.use_hyperstack = True
        self.hyperstack = None

    def set_file(self, filename: str):
        self.filename = filename

        img_dir, img_name = os.path.split(self.filename)

        if self.out is None:
            self.out = os.path.splitext(img_name)[0]
            self.out = os.path.join(img_dir, self.out)

        return self

    def set_output_dir(self, out: str):
        self.out = out
        return self

    def set_threshold(self, threshold: int):
        self.threshold = threshold
        return self

    def set_tolerance(self, tolerance: int):
        self.tolerance = tolerance
        return self

    def force_redo_on(self):
        self.force_redo = True
        return self

    def force_redo_off(self):
        self.force_redo = False
        return self

    def set_force_redo(self, force: bool):
        self.force_redo = force
        return self

    def quality_on(self):
        self.quality = True
        return self

    def quality_off(self):
        self.quality = False
        return self

    def set_quality(self, quality: bool):
        self.quality = quality
        return self

    def asynchronous_on(self):
        self.asynchronous = True
        return self

    def asynchronous_off(self):
        self.asynchronous = False
        return self

    def set_asynchronous(self, asynchronous: bool):
        self.asynchronous = asynchronous
        return self

    def hyperstack_on(self):
        self.use_hyperstack = True
        return self

    def hyperstack_off(self):
        self.use_hyperstack = False
        return self

    def set_hyperstack(self, use_hyperstack: bool):
        self.use_hyperstack = use_hyperstack
        return self

    def _plot(self):

        fig = plt.figure(0, dpi=300)
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(212)
        # fig, ax = plt.subplots(1, 2)

        plot_segmentation(self.neurons, ax=ax1)

        ax1.set_title('Segmentation')
        ax1.set_xlabel('X [px]')
        ax1.set_ylabel('Y [px]')

        swcs = []
        for neuron in self.neurons:
            swcs.append(neuron.swc)

        ax2.set_title('Structure')
        plot_swcs(swcs, ax=ax2, units=self.hyperstack.voxel_unit_str)


        ax3.set_title('Dynamics')
        all_intensities = []
        for neuron in self.neurons:
            all_intensities.append(neuron.intensities)

        all_intensities = np.array(all_intensities)
        im = ax3.imshow(all_intensities,
                        aspect='auto',
                        interpolation=None,
                        extent=[0, self.hyperstack.frames * self.hyperstack.period, 1, len(self.neurons) + 1])
        fig.colorbar(im)

            #ax3.plot(tt, neuron.intensities, label=neuron.num)


        ax3.set_xlabel(f't [{self.hyperstack.period_unit_str}]')
        ax3.set_ylabel('Neuron')
        # plt.legend(loc='best')

        fig.show()

    def _must_read_segmentation_file(self):
        return (
                os.path.exists(self.out)
                and not self.force_redo
                and any([check_long_ext(fname, RIVULET_2_TREE_IMG_EXT) for fname in os.listdir(self.out)])
        )

    def _read_segmentation_from_file(self):
        self.neurons = []
        for fname in os.listdir(self.out):
            if check_long_ext(fname, RIVULET_2_TREE_IMG_EXT):
                loc = os.path.join(self.out, fname)
                image = loadimg(loc, 1)
                self.neurons.append(Neuron(image, img_fname=loc, num=len(self.neurons)))
        print(f'Loaded {len(self.neurons)} neuron images from file.')

        self.hyperstack = HyperStack().from_file(self.filename, metadata_only=True)

    def _segment(self):
        ################ LOAD IMAGE AND METADATA #################
        self.hyperstack = HyperStack().from_file(self.filename)

        ######## CREATE PROJECTIONS FOR TRACING AND VOLTAGE IMAGE DATA ANALYSIS ##########
        spatial_data = self.hyperstack.t_project(mode='MAX')

        # Reload without image
        self.hyperstack = None
        self.hyperstack = HyperStack().from_file(self.filename, metadata_only=True)

        # Create a new directory next to the input file for the SWC outputs
        if not os.path.exists(self.out):
            os.mkdir(self.out)

        neuronsegmentor = NeuronSegmentor(spatial_data, threshold=self.threshold, tolerance=self.tolerance)
        self.neurons = neuronsegmentor.neurons

        # DEBUG PLOTS
        # neuronsegmentor.plot()
        # neuronsegmentor.plot_full_segmentation()

    def _write_segmentation_to_file(self):
        for ii, neuron in enumerate(self.neurons):
            # Store images
            img_fname = 'neuron_{:04d}{}'.format(ii, RIVULET_2_TREE_IMG_EXT)
            loc = os.path.join(self.out, img_fname)
            neuron.img_fname = loc
            sitk.WriteImage(neuron.img, neuron.img_fname)
        print(f'Segmented image into {len(self.neurons)} neurons.')

    @staticmethod
    def _trace_single(neuron, threshold, speed, quality, force_retrace, voxel_size):
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

            swc.apply_scale(voxel_size)

            swc.save(neuron.swc_fname)
            neuron.add_SWC(swc)

        return neuron

    def _trace_all(self):
        if self.asynchronous:
            with Pool(processes=os.cpu_count() - 1) as pool:
                result_buffers = []
                for neuron in self.neurons:
                    result = pool.apply_async(self._trace_single,
                                              (neuron,
                                               self.threshold,
                                               self._speed,
                                               self.quality,
                                               self.force_redo,
                                               self.hyperstack.voxel_size))
                    result_buffers.append(result)

                self.neurons = [result.get() for result in result_buffers]
        else:
            result_buffers = []
            for neuron in self.neurons:
                result_buffers.append(self._trace_single(neuron,
                                                         self.threshold,
                                                         self._speed,
                                                         self.quality,
                                                         self.force_redo,
                                                         self.hyperstack.voxel_size))

            self.neurons = result_buffers

    @staticmethod
    def _get_voltage_single(neuron: Neuron, hyperstack: HyperStack, force_redo: bool, window_scale: int = 5):
        neuron.i_fname = '{}{}'.format(neuron.img_fname.split(RIVULET_2_TREE_IMG_EXT)[0], '.npy')

        if os.path.exists(neuron.i_fname) and not force_redo:
            neuron.intensities = np.load(neuron.i_fname)
            print(f'Neuron ({neuron.num})\t --Loaded trace from disk')

        else:
            soma_centroid = np.divide(neuron.swc._data[0, 2:5], hyperstack.voxel_size) # XYZ of soma in cleaned SWC

            # TODO: Nice implementation for scaling section radii based on voxel size
            # This would likely involve figuring out where the line spanning from childID to parentID is pointing to
            # correctly scale the radius with an appropriate weight for each direction
            radius = neuron.swc._data[0, 5] / np.min(hyperstack.voxel_size) * window_scale
            print(f'Neuron ({neuron.num})\t --Getting intensity trace from a radius {radius:.0f} px from the soma.')

            x_size, y_size, z_size, frames = hyperstack.GetSize()

            mask = sitk.Image((x_size, y_size, z_size), hyperstack.GetPixelID())

            idx = mask.TransformPhysicalPointToIndex(soma_centroid)
            mask[idx] = 1

            bin_dil_filt = sitk.BinaryDilateImageFilter()
            bin_dil_filt.SetKernelRadius(int(radius))
            bin_dil_filt.SetKernelType(sitk.sitkBall)
            mask = bin_dil_filt.Execute(mask)

            # plt.imshow(flatten(mask))
            # plt.show()

            intensities = np.zeros(frames)

            for ii in range(frames):
                volume = hyperstack[:, :, :, ii]
                volume = volume * mask
                intensities[ii] = np.max(sitk.GetArrayFromImage(volume))

            neuron.intensities = intensities
            np.save(neuron.i_fname, intensities)

        return neuron

    def _get_voltage_all(self):
        hyperstack = HyperStack().from_file(self.filename)

        if self.asynchronous:
            with Pool(processes=os.cpu_count() - 1) as pool:
                result_buffers = []
                for neuron in self.neurons:
                    result = pool.apply_async(self._get_voltage_single,
                                              (neuron, hyperstack, self.force_redo))
                    result_buffers.append(result)

                results = [result.get() for result in result_buffers]
        else:
            results = []
            for neuron in self.neurons:
                results.append(self._get_voltage_single(neuron, hyperstack, self.force_redo))

        self.neurons = results


    def execute(self):

        if self.threshold is not None and not isinstance(self.threshold, (int, float, str)):
            raise TypeError(
                'Expected threshold to be either of type str, specifiying an automatic thresholding method \n',
                f'or a number, specifying the threshold value, instead got {type(self.threshold)}')

        # self._read_metadata()

        if self._must_read_segmentation_file():
            self._read_segmentation_from_file()
            # self._read_neurons_from_file()
        else:
            self._segment()
            self._write_segmentation_to_file()
            # self._write_neurons_to_file()

        self._trace_all()
        # self.get_voltage(file, results, asynchronous=False)

        self._get_voltage_all()

        self._plot()

        return self.neurons

