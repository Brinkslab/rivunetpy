import copy
from multiprocessing import Process, Manager

from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from SimpleITK.SimpleITK import Image

from rivuletpy.utils.plottools import flatten
from rivuletpy.utils.metrics import euclidean_distance


def max_between_stacks(*images):
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

def eval_hessian_scale(img: Image, img_list: list, scale, dimension: int, scaled_to_eval = bool) -> Image:
    frangi_filter = sitk.ObjectnessMeasureImageFilter()
    # frangi_filter.SetGamma(1)
    frangi_filter.SetAlpha(0.75)
    frangi_filter.SetBeta(0.75)
    frangi_filter.SetBrightObject(True)
    frangi_filter.SetScaleObjectnessMeasure(scaled_to_eval)
    frangi_filter.SetObjectDimension(dimension)

    gaussian_filter = sitk.DiscreteGaussianImageFilter()
    gaussian_filter.SetUseImageSpacing(False)

    gaussian_filter.SetVariance(int(scale ** 2))  # Sigma = Var^2
    img_blurred = gaussian_filter.Execute(img)
    img_blurred = sitk.Cast(img_blurred, sitk.sitkFloat32)

    img_list.append(frangi_filter.Execute(img_blurred))

def hessian_filter(img, scales, dimension=0, scaled_to_eval=False, parallel=False):
    if parallel:
        with Manager() as manager:
            results = manager.list()
            pps = []

            for scale in scales:
                pp = Process(target=eval_hessian_scale, args=(img, results, scale, dimension, scaled_to_eval))
                pp.start()
                pps.append(pp)

            for pp in pps:
                pp.join()

            # When done with all processes
            results = list(results)

    else:
        results = []

        for scale in scales:
            eval_hessian_scale(img, results, scale, dimension, scaled_to_eval)

    return max_between_stacks(results)

# def hessian_filter(img, scales, dimension=0, scaled_to_eval=False):
#     # TODO: Make this multi-threaded/parallelized
#
#     frangi_filter = sitk.ObjectnessMeasureImageFilter()
#     # frangi_filter.SetGamma(1)
#     frangi_filter.SetAlpha(0.75)
#     frangi_filter.SetBeta(0.75)
#     frangi_filter.SetBrightObject(True)
#     frangi_filter.SetScaleObjectnessMeasure(scaled_to_eval)
#     frangi_filter.SetObjectDimension(dimension)
#
#     gaussian_filter = sitk.DiscreteGaussianImageFilter()
#     gaussian_filter.SetUseImageSpacing(False)
#
#
#
#     return max_between_stacks(images)


def get_max_scale(binary):
    distance_transform = sitk.SignedMaurerDistanceMapImageFilter()
    # distance_transform.SetUseImageSpacing(False)
    distance_transform.SetInsideIsPositive(True)
    # distance_transform.SetBackgroundValue(1)
    distance_transform.SetSquaredDistance(False)
    distance_img = distance_transform.Execute(binary)

    max_filter = sitk.MinimumMaximumImageFilter()
    _ = max_filter.Execute(distance_img)

    return int(max_filter.GetMaximum())


def prune_seeds(seeds, radius):
    valid_seeds = copy.copy(seeds)
    init_seeds = len(valid_seeds)  # Initial number of seeds

    ii = 0
    while True:
        if ii == len(valid_seeds):
            break

        seed = valid_seeds[ii]
        kill_indices = []
        for jj, compared_seed in enumerate(valid_seeds):
            if ii == jj:
                pass
            else:
                delta = euclidean_distance(seed, compared_seed)
                if delta < radius * 2:  # Scale is a radius
                    kill_indices.append(jj)
        valid_seeds = np.delete(valid_seeds, kill_indices, axis=0)

        ii += 1

    return valid_seeds


def get_seeds(img, scale):
    blobs = hessian_filter(img, [scale], dimension=0, scaled_to_eval=True)

    blobs = sitk.RescaleIntensity(blobs, 0, 65535)  # 0-65535

    blobs = sitk.Cast(blobs, sitk.sitkUInt16)

    gauss_filter = sitk.DiscreteGaussianImageFilter()
    gauss_filter.SetUseImageSpacing(False)
    gauss_filter.SetVariance(scale ** 2)  # Var is Sigma^2
    gauss_filter.SetMaximumKernelWidth(500)
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

    return prune_seeds(seeds, scale)


class NeuronSegmentor:

    def __init__(self, img, save=False):
        self.save = save

        self.img = img
        self.PixelID = self.img.GetPixelID()

        threshold_filter = sitk.OtsuThresholdImageFilter()
        threshold_filter.SetInsideValue(1)
        threshold_filter.SetOutsideValue(0)

        threshold_filter.Execute(self.img)
        self.threshold = threshold_filter.GetThreshold()
        self.binary = self.img > self.threshold

        with tqdm(total=9) as pbar:
            self.soma_scale = self.get_soma_scale()
            pbar.update(1)

            self.soma_seeds = self.get_soma_seeds()
            pbar.update(1)

            self.soma_mask = self.get_soma_mask()
            pbar.update(1)

            self.neurite_img = self.get_neurite_img()
            pbar.update(1)

            self.neurite_scale = self.get_neurite_scale()
            pbar.update(1)

            self.neurite_frangi = self.frangi_filter()
            pbar.update(1)

            self.composite_image = self.make_composite_image()
            pbar.update(1)

            self.regions, self.region_labels = self.get_regions()
            pbar.update(1)

            self.neuron_images = self.make_neuron_images()
            pbar.update(1)

    def __str__(self):
        return (f'{len(self.neuron_images)}Neuron(s) with \n\tSoma Scale = {self.soma_scale}\n'
                f'\tNeurite Scale = {self.neurite_scale}')

    def get_soma_scale(self):
        return get_max_scale(self.binary)

    def get_soma_seeds(self):
        return get_seeds(self.img, self.soma_scale)

    def get_soma_mask(self, print_time=False):
        mask = sitk.ConfidenceConnected(self.img, seedList=self.soma_seeds.tolist(),
                                        numberOfIterations=1,
                                        multiplier=1,
                                        initialNeighborhoodRadius=1,
                                        replaceValue=1)
        return mask

    def get_neurite_img(self, dilate=True):
        mask = self.soma_mask

        if dilate is not None:
            if type(dilate) is bool:
                radius = self.soma_scale

            elif type(dilate) in (int, bool):
                radius = dilate
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

    def get_neurite_scale(self):
        threshold_filter = sitk.MaximumEntropyThresholdImageFilter()
        threshold_filter.SetInsideValue(1)
        threshold_filter.SetOutsideValue(0)
        threshold_filter.Execute(self.neurite_img)
        threshold = threshold_filter.GetThreshold()
        binary = self.neurite_img > threshold

        neurite_scale = get_max_scale(binary)
        if neurite_scale < 1:
            neurite_scale = 1

        return neurite_scale

    def frangi_filter(self):
        if self.neurite_scale == 1:
            neurite_scale = 2
        else:
            neurite_scale = self.neurite_scale

        if neurite_scale > self.soma_scale / 2:
            scales = [neurite_scale]  # For when scale and soma scale too close together. np.arange will return []
        else:
            scales = np.arange(neurite_scale, self.soma_scale / 2).astype(int)

        frangi = hessian_filter(self.img, scales, dimension=1)

        frangi = sitk.RescaleIntensity(frangi, 0, 65535)  # 0-255

        return sitk.Cast(frangi, sitk.sitkUInt16)

    def make_composite_image(self):
        mask = self.soma_mask

        mask = sitk.Cast(mask, sitk.sitkUInt16)

        label_image = sitk.ConnectedComponent(mask)
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.Execute(label_image, self.neurite_frangi)

        soma_sticker = sitk.Image(self.img.GetSize(), sitk.sitkUInt16)

        for label in stats.GetLabels():
            mean_I = stats.GetMean(label)
            max_I = stats.GetMaximum(label)

            region = label_image == label

            dilate_filter = sitk.BinaryDilateImageFilter()
            dilate_filter.SetKernelRadius(self.neurite_scale)
            dilate_filter.SetForegroundValue(1)

            region = dilate_filter.Execute(region)

            soma_sticker += sitk.Cast(region, sitk.sitkUInt16) * mean_I

        inverted_mask = sitk.Cast(mask == 0, sitk.sitkUInt16)

        return self.neurite_frangi * inverted_mask + soma_sticker

    def get_regions(self):

        threshold_filter = sitk.TriangleThresholdImageFilter()
        threshold_filter.SetInsideValue(1)
        threshold_filter.SetOutsideValue(0)
        threshold_filter.Execute(self.composite_image)

        threshold = threshold_filter.GetThreshold()

        mask = sitk.ConnectedThreshold(self.composite_image,
                                       seedList=self.soma_seeds.tolist(),
                                       lower=threshold,
                                       upper=65535)

        # global_mask = max_between_stacks(mask, self.soma_mask)

        if self.save:
            self.orig_mask = mask

        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetKernelRadius(self.neurite_scale)
        dilate_filter.SetForegroundValue(1)

        mask = dilate_filter.Execute(mask)

        if self.save:
            self.dil_mask = mask

        components = sitk.ConnectedComponent(mask)

        dist_filter = sitk.SignedDanielssonDistanceMapImageFilter()
        dist_filter.Execute(components)
        voronoi = dist_filter.GetVoronoiMap()

        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.Execute(voronoi, self.img)

        return voronoi, stats.GetLabels()

    def make_neuron_images(self):
        neuron_images = []
        for label in self.region_labels:
            region = self.regions == label
            image = self.img * sitk.Cast(region, self.PixelID)
            neuron_images.append(image)

        return neuron_images

    def plot(self):
        fig = plt.figure(figsize=(10, 5), dpi=200)

        ax = fig.add_subplot(1, 2, 1)
        shape = self.img.GetSize()
        plt.title(f'{len(self.soma_seeds)} Unique seeds')
        soma_seeds = np.array(self.soma_seeds)
        plt.imshow(flatten(self.img),
                   cmap='gray',
                   interpolation='none',
                   extent=[0, shape[0], 0, shape[1]])
        x = soma_seeds[:, 0]
        y = shape[0] - soma_seeds[:, 1]
        plt.plot(x, y, marker='x', markersize=20, linestyle='none')
        plt.axis('square')
        plt.axis('off')

        ax = fig.add_subplot(1, 2, 2)
        shape = self.regions.GetSize()
        plt.title('Labeled neurons')
        plt.imshow(flatten(self.regions),
                   cmap='nipy_spectral',
                   interpolation='none',
                   extent=[0, shape[0], 0, shape[1]])
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        for image in self.neuron_images:
            fig = plt.figure()
            plt.imshow(flatten(image), cmap='gray')
            plt.show()

    def plot_full_segmentation(self):

        fig = plt.figure(figsize=(20, 7), dpi=200)

        ax = fig.add_subplot(2, 4, 1)
        plt.title(f'Original image, soma size {self.soma_scale} px')
        plt.imshow(flatten(self.img), cmap='gray', interpolation='none')
        plt.colorbar()
        plt.axis('off')

        ax = fig.add_subplot(2, 4, 2)
        shape = self.img.GetSize()
        plt.title(f'{len(self.soma_seeds)} Unique seeds')
        soma_seeds = np.array(self.soma_seeds)
        plt.imshow(flatten(self.img), cmap='gray', interpolation='none', extent=[0, shape[0], 0, shape[1]])
        x = soma_seeds[:, 0]
        y = shape[0] - soma_seeds[:, 1]
        plt.plot(x, y, color='red', marker='x', markersize=20, linestyle='none')
        plt.colorbar()
        plt.axis('square')
        plt.axis('off')

        ax = fig.add_subplot(2, 4, 3)
        plt.title('Soma mask')
        plt.imshow(flatten(self.soma_mask), cmap='gray', interpolation='none')
        plt.colorbar()
        plt.axis('off')

        ax = fig.add_subplot(2, 4, 4)
        plt.title(f'Neurites only, size={self.neurite_scale} px')
        plt.imshow(flatten(self.neurite_img), cmap='gray', interpolation='none')
        plt.colorbar()
        plt.axis('off')

        ax = fig.add_subplot(2, 4, 5)
        plt.title('Frangi')
        plt.imshow(flatten(self.neurite_frangi), cmap='gray', interpolation='none')
        plt.colorbar()
        plt.axis('off')

        ax = fig.add_subplot(2, 4, 6)
        plt.title('Composite image')
        plt.imshow(flatten(self.composite_image), cmap='gray', interpolation='none')
        plt.axis('off')
        plt.colorbar()

        ax = fig.add_subplot(2, 4, 7)
        plt.title('Regions')
        plt.imshow(flatten(self.regions), cmap='nipy_spectral', interpolation='none')
        plt.colorbar()
        plt.axis('off')

        plt.tight_layout()
        plt.show()
