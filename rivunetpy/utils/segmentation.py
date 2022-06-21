"""Segments 3D stacks of neuron micrographs into individual images for tracing.

Rivuletpy 0.2 does not support tracing multiple neurons. This module is intended as a pre-processor for the
Rivuletpy tracer that segments a 3D stack containing mulitple neurons to indivudual images containing just
one neuron.

  Typical usage example:

   neurons = NeuronSegmentor(image)
   neuron_images = neurons.neuron_images

"""

import copy
from typing import Union
from multiprocessing import Process, Manager

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from SimpleITK.SimpleITK import Image
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh

from rivunetpy.utils.cells import Neuron
from rivunetpy.utils.plottools import flatten
from rivunetpy.utils.metrics import euclidean_distance


def max_between_stacks(*images) -> Image:
    """Performs a element-wise maximum projection between any number of 3D stack.

    Args:
        *images: Collection of sitk.Image that all have the same shape (N x M x L) and type.

    Returns:
        Image: A sitk.Image of shape (N x M x L) where each element equals the maximum at that element in any
        of the input stacks.

    Raises:
        AssertionError: The input images did not have the same shape or type.
    """
    if len(images) == 1 and type(images[0]) is list:  # Allow for both list and multiple args input
        images = images[0]

    img_shapes = []
    img_types = []
    img: Image
    for img in images:
        assert type(img) is Image, ('Input image should by of the SimpleITK.SimpleITK.Image type, \n'
                                    f'instead is {type(img)}')
        img_shapes.append(img.GetSize())
        img_types.append(img.GetPixelID())

    assert len(set(img_shapes)) == 1, ('All the input images should have the same shape\n'
                                       f'instead they have {",".join([str(sh) for sh in img_shapes])}')
    assert len(set(img_types)) == 1, 'All the input images should have the same types'

    img_shape = img_shapes[0]
    stack_shape = (len(images), img_shape[-1], *img_shape[:-1])

    stack = np.zeros(stack_shape)

    for ii, img in enumerate(images):
        arr_buffer = sitk.GetArrayFromImage(img)
        stack[ii] = arr_buffer

    stack_max = np.max(stack, axis=0)
    stack_max = sitk.GetImageFromArray(stack_max)

    stack_max = sitk.Cast(stack_max, int(img_types[0]))
    return stack_max


def eval_hessian_scale(img: Image, img_list: list, scale: Union[int, float],
                       dimension: int, scaled_to_eval: bool, normalized: bool) -> Image:
    """Applies a hessian-type filter on a 3D stack.

     Used by ``hessian_filter``. When complete, the result is appended to ``img_list``.

    Args:
        img (Image): The image to which the hessian-type filter is applied.
        img_list (list): A list to which filtered images are appended. Intensities are stored as a 32-bit floating
          point value.
        scale (int): The scale over which the hessian is evaluated.
        dimension (int): Dimensionality of an object that corresponds to high intensities in the output.
          0 corresponds to blob-like objects. 1 corresponds to linear-like objects.
        scaled_to_eval (bool): If set to True, the intensities of the output image are scaled to the size of
          the largest eigenvalue of the hessian.
        normalized (bool): If set to True, all the intensities of the output image are scaled to a range from 0-1.
    """
    frangi_filter = sitk.ObjectnessMeasureImageFilter()
    # frangi_filter.SetGamma(1)
    # frangi_filter.SetAlpha(0.5)
    # frangi_filter.SetBeta(0.5)
    frangi_filter.SetBrightObject(True)
    frangi_filter.SetScaleObjectnessMeasure(scaled_to_eval)
    frangi_filter.SetObjectDimension(dimension)

    gaussian_filter = sitk.DiscreteGaussianImageFilter()
    gaussian_filter.SetMaximumKernelWidth(1000)
    gaussian_filter.SetUseImageSpacing(False)

    gaussian_filter.SetVariance(int(scale ** 2))  # Sigma = Var^2
    img_blurred = gaussian_filter.Execute(img)
    img_blurred = sitk.RescaleIntensity(img_blurred, 0, 65535)
    img_blurred = sitk.Cast(img_blurred, sitk.sitkFloat32)

    result = frangi_filter.Execute(img_blurred)

    x = result.GetDimension()

    if normalized:
        result = sitk.RescaleIntensity(result, 0, 1)

    img_list.append(result)


def hessian_filter(img: Image, scales: list, dimension: int = 0,
                   scaled_to_eval=False, normalized=False, parallel=False) -> Image:
    """Applies a (multi-scale) hessian-like filtering operation on a 3D stack.

    Transforms an image to an input image where each pixel represents an object-ness measure at
    that location. By setting the dimension of the objects to be enhanced, both blob-like (0D) objects, and
    vessel-like objects (1D) can be recovered.

    Args:
        img (Image): The image to which the hessian-type filter is applied
        scales (list): List of integer scales at which features will be filtered.
        dimension (int): Type of features that will be filtered. The filter enhances blob-like structures when
          ``dimension`` is set to 0. Vessel like structures are enhanced when set to 1.
        scaled_to_eval (bool): If set to True, the object-ness measure at each location in the image is
          scaled by the magnitude of the largest absolute eigenvalue.
        normalized (bool): If set to True, all the intensities of the output image are scaled to a range from 0-1.
        parallel (bool): If set to True, each scale in ``scales`` will be processed on a different thread. Doing so
          might incur an extra penalty in processing time due to the initialization steps. Speed gains are expected
          for large sets of scales.

    Returns:
        Image: An image where each pixel in the stack has an intensity that corresponds to how similar it is in
          shape to the object (blob, vessel) specified by ``dimension``.

    Todo:
        * Benchmark parallelization

    """
    if len(scales) > 1:
        normalized = True

    if parallel:
        with Manager() as manager:
            results = manager.list()
            pps = []

            for scale in scales:
                pp = Process(target=eval_hessian_scale,
                             args=(img, results, scale, dimension, scaled_to_eval, normalized))
                pp.start()
                pps.append(pp)

            for pp in pps:
                pp.join()

            # When done with all processes
            results = list(results)

    else:
        results = []

        for scale in scales:
            eval_hessian_scale(img, results, scale, dimension, scaled_to_eval, normalized)

    return max_between_stacks(results)


def find_max_scale(binary: Image) -> int:
    """Finds the approximate scale of the largest object in a binary image.

    Args:
        binary (Image): Binary image.

    Returns:
        Integer representing a radius-like measure for the size of the largest object in an image (pixel units).
    """

    distance_transform = sitk.SignedMaurerDistanceMapImageFilter()
    # distance_transform.SetUseImageSpacing(False)
    distance_transform.SetInsideIsPositive(True)
    # distance_transform.SetBackgroundValue(1)
    distance_transform.SetSquaredDistance(False)
    distance_img = distance_transform.Execute(binary)

    max_filter = sitk.MinimumMaximumImageFilter()
    _ = max_filter.Execute(distance_img)

    return int(max_filter.GetMaximum())


def prune_points(points: list, radius: Union[int, float]) -> np.ndarray:
    """Takes a list of coordinates and makes every point unique within a certain radius.

    Of the points in a N-dimensional space, only those that lie a certain radius apart are kept. In doing so,
    the points are found that uniquely inhabit a space within a tolerance specified by a radius.

    Args:
        points: List of points out of which to remove only the unique points.
        radius: Radius in the N-dimensional space that at a minimum must separate two "unique" points.

    Returns:
        np.ndarray: A ``numpy`` array of unique points.
    """
    valid_points = copy.copy(points)

    ii = 0
    while True:
        if ii == len(valid_points):
            break

        point = valid_points[ii]
        kill_indices = []
        for jj, compared_point in enumerate(valid_points):
            if ii == jj:
                pass
            else:
                delta = euclidean_distance(point, compared_point)
                if delta < radius * 2:  # Scale is a radius
                    kill_indices.append(jj)
        valid_points = np.delete(valid_points, kill_indices, axis=0)

        ii += 1

    return valid_points


def get_seeds(img: Image, scale: int, threshold: float = 0.20, exclude_border_dist: int = None) -> np.ndarray:
    """Gives the coordinates of blobs of a certain scale in a 3D stack.

    Retrieves the points at which blobs of a specified scale lie in a 3D stack image. Each blob is assigned strictly
    one point. These points do not necessarily lie on the centroid of a volume in that image. Useful for generating
    seed points for segmenting images.

    Args:
        img: Input image from which to extract seeds.
        scale: Scale of blobs which are assigned seed points.
        threshold: Tolerance for differently-sized somata

    Returns:
        np.ndarray: A ``numpy`` array of points, each lying on one blob in a location that is a-specific to
          the geometry of that blob.
    """

    if img.GetDimension() == 3:

        scales = np.arange(int(scale * (1 - threshold)), int(scale * (1 + threshold)))

        blobs = hessian_filter(img, scales, scaled_to_eval=False, dimension=0)

        blobs = sitk.RescaleIntensity(blobs, 0, 65535)  # 0-65535

        if exclude_border_dist is not None:
            rad = exclude_border_dist
            # Only take off XY border, Z slice too thin
            blobs[:rad, :, :] = 0
            blobs[-rad:, :, :] = 0
            blobs[:, :rad, :] = 0
            blobs[:, -rad:, :] = 0

        blobs = sitk.Cast(blobs, sitk.sitkUInt16)

        gauss_filter = sitk.DiscreteGaussianImageFilter()
        gauss_filter.SetUseImageSpacing(False)
        gauss_filter.SetVariance(scale ** 2)  # Var is Sigma^2
        gauss_filter.SetMaximumKernelWidth(1000)
        blobs = gauss_filter.Execute(blobs)

        threshold_filter = sitk.MaximumEntropyThresholdImageFilter()
        threshold_filter.SetInsideValue(1)
        threshold_filter.SetOutsideValue(0)
        threshold_filter.Execute(blobs)

        threshold = threshold_filter.GetThreshold()
        blobs = blobs > threshold

        label_image = sitk.ConnectedComponent(blobs)
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.Execute(label_image, img)

        seeds = []
        for label in stats.GetLabels():
            cent = np.array(stats.GetCentroid(label)).astype(int).tolist()
            seeds.append(tuple(cent))

    elif img.GetDimension() == 2:
        seeds = blob_dog(sitk.GetArrayFromImage(img), min_sigma=scale * 0.75, max_sigma=scale * 1.25, threshold=.20)
        seeds = seeds[:, [1, 0]].astype(int)

    else:
        raise RuntimeWarning('Incorrect dimension of image during seed finding. '
                             f'\nImage has a dimension {img.GetDimension()}.'
                             '\nTry using an input image with a (spatial) dimensionality 2 or 3')

    return prune_points(seeds, scale)


class NeuronSegmentor:
    """Pipeline for segmenting images of multiple neurons into images containing only single neurons.

    Attributes:
        img (Image): Input image
        threshold (float): Threshold used in the initial processing steps.
        aggressiveness (float): How agressive the thresholding is when adding pixels to the neuron regions.
        binary (Image): Binarized version of the input image
        soma_scale (int): Scale of the largest soma in the image.
        soma_seeds (np.ndarray): Locations of all the individual somata in the image.
        neurite_scale (int): The approximate scale of the neurites in the image.
        regions (Image): An image where each region in which a single neuron lies is labeled by a unique intensity value.
        neurons: Images of the individual neurons.
    """

    def __init__(self, img: Image, threshold: Union[int, float] = None, tolerance=0.15):
        """Segment an image of multiple neurons.

        A progress bar is shown to indicate the approximate progress.

        Args:
            img: Input image that will be segmented.
            threshold (float, optional): Optional manual threshold setting. If no threshold is passed,
              Maximum Entropy thresholding will be used for the initial thresholding of the image.

        Raises:
            ValueError: If the threshold is not a number.
        """
        print('Starting segmentation')
        self.img = img
        self.PixelID = self.img.GetPixelID()

        self.seed_tolerance = tolerance

        if threshold is None:
            threshold_filter = sitk.MaximumEntropyThresholdImageFilter()
            threshold_filter.SetInsideValue(1)
            threshold_filter.SetOutsideValue(0)

            threshold_filter.Execute(self.img)
            self.threshold = threshold_filter.GetThreshold()
        elif type(threshold) in (int, float):
            self.threshold = threshold
        else:
            raise ValueError(f'Threshold value should be of type int or float, instead got {type(threshold)}')

        self.binary = self.img > self.threshold
        self.binary_closed = None

        self.components = None

        with tqdm(total=100) as pbar:
            self.soma_scale = self.__find_soma_scale()
            pbar.update(1)  # Update values weighted by function duration

            self.soma_seeds = self.__find_soma_seeds()
            pbar.update(16)

            self.__soma_cover = self.__make_soma_overfit_cover()
            pbar.update(1)

            self.__neurite_img = self.__make_neurite_img()
            pbar.update(2)

            self.neurite_scale = self.__find_neurite_scale()
            pbar.update(1)

            self.__neurite_frangi = self.__apply_frangi_filter()
            pbar.update(7)

            self.__composite_image = self.__make_segmenting_image()
            pbar.update(11)

            self.regions, self.__region_labels = self.__find_regions()
            pbar.update(50)

            self.neurons = self.__make_neuron_images()
            pbar.update(11)

    def __str__(self):
        return (f'{len(self.__region_labels)} Neuron(s) with \n\tSoma Scale = {self.soma_scale}\n'
                f'\tNeurite Scale = {self.neurite_scale}')

    def __find_soma_scale(self) -> int:
        """Uses the binary image of the soma to estimate soma scale.

        Only works if the original image was correctly binarized.

        Designed to also work with "hollow" somata images due to membrane expression of GEVIs.
        First flattens image to a 2D projection to estiamte scale. This assumes that the cells are in culture
        and thus have soma that are larger in XY than Z. A second pass is then done in 3D using a binary image
        on which morphological closing is performed. The estimated scale is used for the closing kernel size.

        Returns:
            int: An integer that is approximately equal to the radius of the soma in pixel units.
        """
        flat_bin = flatten(self.binary, as_sitk=True)

        distance_transform = sitk.SignedMaurerDistanceMapImageFilter()
        # distance_transform.SetUseImageSpacing(False)
        distance_transform.SetInsideIsPositive(True)
        # distance_transform.SetBackgroundValue(1)
        distance_transform.SetSquaredDistance(False)
        distance_img = distance_transform.Execute(flat_bin)

        max_filter = sitk.MinimumMaximumImageFilter()
        _ = max_filter.Execute(distance_img)

        scale_guess_2D = int(max_filter.GetMaximum())

        clos_filt = sitk.BinaryMorphologicalClosingImageFilter()

        clos_filt.SetKernelRadius(int(scale_guess_2D))

        self.binary_closed = clos_filt.Execute(self.binary)

        scale_guess_3D = find_max_scale(self.binary_closed)

        return max(scale_guess_3D, scale_guess_2D)

    def __find_soma_seeds(self) -> np.ndarray:
        """Finds unique points on the soma that can act as seeds for segmentation.

        Assumes the somata are blob like and uses this property to filter for blob-like structures unique
        points at each blob. Other blob-like structures that may not be soma are also included in this list as the
        result of the nature of this algorithm. It is assumed that these will be dealt with later.

        Returns:
            np.ndarray: A ``numpy`` array containing the locations of the somata in pixel units.
        """
        return get_seeds(self.img, self.soma_scale, threshold=self.seed_tolerance, exclude_border_dist=self.soma_scale)

    def __make_soma_overfit_cover(self) -> Image:
        """Create a mask that is guaranteed to cover the somata in an image.

        Creates a binary mask that can be used to mask off the somata from the input image. This mask is an overfit
        and as such, some neurite structure will also be cut off. The purpose of this mask is to use this to
        pre-process the original image (remove somata) in order to find the neurite scale using a binarized image
        of the neurites.

        Returns:
            Image: A mask that (over-) covers the somata in the input image.

        """
        # mask = sitk.ConfidenceConnected(self.img, seedList=self.soma_seeds.tolist(),
        #                                 numberOfIterations=1,
        #                                 multiplier=1,
        #                                 initialNeighborhoodRadius=1,
        #                                 replaceValue=1)

        size = self.img.GetSize()
        marker = sitk.Image(*size, sitk.sitkUInt8)

        for point in self.soma_seeds.tolist():
            idx = marker.TransformPhysicalPointToIndex(point)
            marker[idx] = 1

        recon_filter = sitk.BinaryReconstructionByDilationImageFilter()
        mask = recon_filter.Execute(marker, self.binary_closed)

        dil_filter = sitk.BinaryDilateImageFilter()
        dil_filter.SetKernelRadius(self.soma_scale * 2)
        marker = dil_filter.Execute(marker)

        return sitk.RescaleIntensity(sitk.Cast(marker * mask, sitk.sitkUInt16), 0, 65535)

    def __make_neurite_img(self, dilate: Union[int, float, bool] = True) -> Image:
        """Create a version of the original images where the somata have been removed.

        Uses a mask generated by ``__make_soma_overfit_cover`` to set the intensities of the soma regions to zero.
        The resultant image should then only contain neurites. To ensure that the somata are entirely removed, the
        removal mask can be enlarged by binary dilation.

        Args:
            dilate: If set to False or None, no dilation whatsoever will take place. If set to True, dilation will take
              place using an automatically-determined kernel size based on the soma scale. If a number is passed,
              dilation will take place using a kernel size that equal to the input number.

        Returns:
            Image: An image containing just the neurites.
        """
        mask = self.__soma_cover

        if dilate is False:
            pass
        elif dilate is None:
            pass
        else:
            if type(dilate) is bool:
                radius = self.soma_scale

            elif type(dilate) in (int, float):
                radius = int(dilate)
            else:
                raise ValueError('Expected dilate to be an integer representing the kernel'
                                 f'Radius for the dilation filter, instead got {dilate}'
                                 'Alternatively, pass True to automatically set size.')

            dilate_filter = sitk.BinaryDilateImageFilter()
            dilate_filter.SetKernelRadius(radius)
            dilate_filter.SetForegroundValue(1)
            mask = dilate_filter.Execute(mask)

        mask = mask == 0  # Invert mask
        mask = sitk.Cast(mask, self.img.GetPixelID())

        return self.img * mask

    def __find_neurite_scale(self) -> int:
        """Use the neurite-only image to retrieve a neurite scale.

        Returns:
            int: The approximate scale of the neurites as a radius in pixel units.
        """
        threshold_filter = sitk.MaximumEntropyThresholdImageFilter()
        threshold_filter.SetInsideValue(1)
        threshold_filter.SetOutsideValue(0)
        threshold_filter.Execute(self.__neurite_img)
        threshold = threshold_filter.GetThreshold()
        binary = self.__neurite_img > threshold

        neurite_scale = find_max_scale(binary)
        if neurite_scale < 1:
            neurite_scale = 1

        return neurite_scale

    def __apply_frangi_filter(self) -> Image:
        """Apply multi-scale frangi filtering on the original image.

        Calculates a range of scales between the neurite scale and half of the soma scale. For each scale in this
        range, a hessian-type filter is applied. The intensities of the resultant image are the object-ness
        measure at that location for that specific scale. For each pixel in the returned image, the highest object-ness
        measure found for all the scales is chosen.

        Returns:
            Image: A 16-bit filtered image with enhanced vessel-like features.
        """
        if self.neurite_scale == 1:
            neurite_scale = 2
        else:
            neurite_scale = self.neurite_scale

        # if neurite_scale > self.soma_scale / 2:
        #     scales = [neurite_scale]  # For when scale and soma scale too close together.
        #     # np.arange will return []
        # else:
        #     scales = np.arange(neurite_scale, self.soma_scale / 2).astype(int)

        scales = np.arange(neurite_scale * 0.75, neurite_scale * 1.25)

        frangi = hessian_filter(self.img, scales, dimension=1)

        frangi = sitk.RescaleIntensity(frangi, 0, 65535)  # 0-255

        return sitk.Cast(frangi, sitk.sitkUInt16)

    def __make_segmenting_image(self) -> Image:
        """Tweak the soma in the frangi-filtered image to allow for seed-based segmentation.

        Pre-processing step to create an image for the final segmentation. Fills the low-intensity
        (low-vessel-object-ness) centers of the frangi-filtered image with the maximum intensity value in the
        neighbourhood.

        Returns:
            Image: A 16-bit modification to the frangi-filtered image with high-intensity somata.
        """

        # mask = self.__soma_cover
        #
        # mask = sitk.Cast(mask, sitk.sitkUInt16)
        #
        # label_image = sitk.ConnectedComponent(mask)
        # stats = sitk.LabelIntensityStatisticsImageFilter()
        # stats.Execute(label_image, self.__neurite_frangi)
        #
        # soma_patch = sitk.Image(self.img.GetSize(), sitk.sitkUInt16)
        #
        # for label in stats.GetLabels():
        #     mean_I = stats.GetMean(label)
        #     max_I = stats.GetMaximum(label)
        #
        #     region = label_image == label
        #
        #     dilate_filter = sitk.BinaryDilateImageFilter()
        #     dilate_filter.SetKernelRadius(self.neurite_scale)
        #     dilate_filter.SetForegroundValue(1)
        #
        #     region = dilate_filter.Execute(region)
        #
        #     soma_patch += sitk.Cast(region, sitk.sitkUInt16) * mean_I
        #
        # inverted_mask = sitk.Cast(mask == 0, sitk.sitkUInt16)
        #
        # return self.__neurite_frangi * inverted_mask + soma_patch

        # result = self.__neurite_frangi
        # clos_filter = sitk.GrayscaleMorphologicalClosingImageFilter()
        # clos_filter.SetKernelRadius(1)
        # for ii in range(self.soma_scale):
        #     result = clos_filter.Execute(self.__neurite_frangi)

        return max_between_stacks(self.__neurite_frangi, self.__soma_cover)

    def __find_regions(self) -> tuple:
        """Find single-neuron regions in the pre-processed composite image.

        Filter that segments the image into regions that contain a single neuron. First performs a seed-based
        segmentation step followed by a connected components step to get a rough segmentation of the individual cells,
        then uses the labeled images to create more expansive volumes containing the neurons.

        Returns:
            tuple: A tuple containing a labeled image and a list of the labels in this image. Each labeled area in the
              image should contain a single neuron.
        """
        # threshold_filter = sitk.LiThresholdImageFilter()
        # threshold_filter.SetInsideValue(1)
        # threshold_filter.SetOutsideValue(0)
        # threshold_filter.Execute(self.__composite_image)
        # threshold = threshold_filter.GetThreshold()
        #
        # mask = sitk.ConnectedThreshold(self.__composite_image,
        #                                           seedList=self.soma_seeds.tolist(),
        #                                           lower=int(threshold * max(2 - self.aggressiveness, 0)),
        #                                           upper=65535)

        threshold_filter = sitk.OtsuThresholdImageFilter()
        threshold_filter.SetInsideValue(1)
        threshold_filter.SetOutsideValue(0)
        threshold_filter.Execute(self.__composite_image)
        threshold = threshold_filter.GetThreshold()

        mask = self.__composite_image > threshold

        bin_close = sitk.BinaryMorphologicalClosingImageFilter()
        bin_close.SetKernelRadius(self.neurite_scale)
        mask = bin_close.Execute(mask)

        ##########################
        size = self.img.GetSize()
        marker = sitk.Image(*size, sitk.sitkUInt16)

        for point in self.soma_seeds.tolist():
            idx = marker.TransformPhysicalPointToIndex(point)
            marker[idx] = 1

        marker = sitk.ConnectedComponent(marker)

        d = sitk.SignedMaurerDistanceMap(mask, insideIsPositive=False, squaredDistance=False, useImageSpacing=False)

        ws = sitk.MorphologicalWatershedFromMarkers(d, marker)

        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.Execute(ws, self.img)

        # dilate_filter = sitk.BinaryDilateImageFilter()
        # dilate_filter.SetKernelRadius(self.soma_scale)
        # dilate_filter.SetForegroundValue(1)
        # mask = dilate_filter.Execute(mask)
        #
        # neighborhood_filter = sitk.NeighborhoodConnectedImageFilter()
        # for seed in self.soma_seeds.tolist():
        #     neighborhood_filter.AddSeed(seed)
        # neighborhood_filter.SetUpper(1)
        # neighborhood_filter.SetLower(1)
        # mask = neighborhood_filter.Execute(mask)

        self.components = ws

        return ws, stats.GetLabels()

        # if self.strict:
        #     stats = sitk.LabelIntensityStatisticsImageFilter()
        #     stats.Execute(self.components, self.img)
        #
        #     return self.components, stats.GetLabels()
        # else:
        #     dist_filter = sitk.SignedDanielssonDistanceMapImageFilter()
        #     dist_filter.Execute(self.components)
        #     voronoi = dist_filter.GetVoronoiMap()
        #
        #     stats = sitk.LabelIntensityStatisticsImageFilter()
        #     stats.Execute(voronoi, self.img)
        #
        #     return voronoi, stats.GetLabels()

    def __make_neuron_images(self) -> list:
        """Create images containing single neurons using the labeled region image.

        Returns:
            list: List of items of cell dataclass, each containing an image of single neuron.
        """
        neurons = []
        for ii, label in enumerate(self.__region_labels):
            region = self.regions == label
            image = self.img * sitk.Cast(region, self.PixelID)
            neurons.append(Neuron(image, num=ii, soma_radius=self.soma_scale))

        return neurons

    def __plot_seeds(self):
        shape = self.img.GetSize()

        soma_seeds = np.array(self.soma_seeds)
        plt.imshow(flatten(self.img),
                   cmap='gray',
                   extent=[0, shape[0], 0, shape[1]])
        x = soma_seeds[:, 0]
        y = shape[0] - soma_seeds[:, 1]
        plt.plot(x, y, color='royalblue', marker='x', markersize=20, linestyle='none')

    def __plot_neuron_images(self):
        cmaps = ['hot', 'bone', 'copper', 'pink']
        num_cmaps = len(cmaps)

        shape = self.regions.GetSize()

        neuron_images = [neuron.img for neuron in self.neurons]

        for ii, image in enumerate(neuron_images[::-1]):
            cmap = matplotlib.cm.get_cmap(cmaps[ii % num_cmaps])

            # Create color map with step in alpha channel
            # Convert threshold to value between 0-1
            step = round(self.threshold / sitk.GetArrayFromImage(self.img).max(), 3)
            aa_cmap = cmap(np.arange(cmap.N))
            aa_cmap[:, -1] = np.linspace(0, 1, cmap.N) > step
            aa_cmap = ListedColormap(aa_cmap)

            plt.imshow(flatten(image),
                       cmap=aa_cmap,
                       extent=[0, shape[0], 0, shape[1]],
                       alpha=1)

    def plot(self):
        """Simple plotting.

        Shows the original image, annotated using the somata locations used as seeds for plotting, along with an image
        with the individually labeled neurons.
        """


        plt.style.use('dark_background')



        fig = plt.figure(figsize=(10, 5), dpi=200)

        ax = fig.add_subplot(1, 2, 1)
        self.__plot_seeds()
        plt.title(f'{len(self.soma_seeds)} Unique seeds')
        plt.axis('square')
        plt.axis('off')

        ax = fig.add_subplot(1, 2, 2)
        self.__plot_neuron_images()
        plt.title(f'{len(self.__region_labels)} Labeled neurons')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def plot_full_segmentation(self, colorbar=False):
        """Plot the entire segmentation process.

          Useful for debugging.
          """
        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        plt.style.use('dark_background')

        fig = plt.figure(figsize=(15, 7), dpi=400)

        ax = fig.add_subplot(2, 4, 1)
        plt.title(f'Original image, soma size: {self.soma_scale} px')
        plt.imshow(flatten(self.img), cmap='gray', interpolation='none')
        if colorbar:
            plt.colorbar()
        plt.axis('off')

        ax = fig.add_subplot(2, 4, 2)
        plt.title('Initial binary pass')
        plt.imshow(flatten(self.binary), cmap='gray', interpolation='none')
        if colorbar:
            plt.colorbar()
        plt.axis('off')

        ax = fig.add_subplot(2, 4, 3)
        self.__plot_seeds()
        plt.title(f'{len(self.soma_seeds)} Unique seeds')
        if colorbar:
            plt.colorbar()
        plt.axis('square')
        plt.axis('off')

        ax = fig.add_subplot(2, 4, 4)
        plt.title('Soma mask')
        plt.imshow(flatten(self.__soma_cover), cmap='gray', interpolation='none')
        if colorbar:
            plt.colorbar()
        plt.axis('off')

        ax = fig.add_subplot(2, 4, 5)
        plt.title(f'Frangi. Neurite size: {self.neurite_scale} px')
        plt.imshow(flatten(self.__neurite_frangi), cmap='gray', interpolation='none')
        if colorbar:
            plt.colorbar()
        plt.axis('off')

        ax = fig.add_subplot(2, 4, 6)
        plt.title('Composite image')
        plt.imshow(flatten(self.__composite_image), cmap='gray', interpolation='none')
        if colorbar:
            plt.colorbar()
        plt.axis('off')

        ax = fig.add_subplot(2, 4, 7)
        plt.title(f'Final regions')
        plt.imshow(flatten(self.components), cmap='nipy_spectral', interpolation='none')
        if colorbar:
            plt.colorbar()
        plt.axis('off')

        ax = fig.add_subplot(2, 4, 8)
        self.__plot_neuron_images()
        plt.title(f'{len(self.__region_labels)} Labeled neurons')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
