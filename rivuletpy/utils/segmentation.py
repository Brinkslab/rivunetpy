import copy
import time

from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from SimpleITK.SimpleITK import Image

from rivuletpy.utils.filtering import apply_threshold
from rivuletpy.utils.plottools import flatten


def get_intensity(img):
    if type(img) is sitk.Image:
        img = sitk.GetArrayFromImage(img)

    return np.sum(img)


def distance(point1, point2):
    return np.sqrt(np.sum(np.square(np.sum(np.array([np.array(point1), -np.array(point2)]), axis=0))))


class NeuronSegmentor:

    def __init__(self, img, save=False):
        self.save = save

        self.img = img
        self.PixelID = self.img.GetPixelID()
        self.binary, self.threshold = apply_threshold(self.img, mthd='Otsu')

        if self.save:
            self.orig_mask = None
            self.dil_mask = None

        self.soma_scale = None
        self.soma_seeds = None
        self.soma_mask = None
        self.neurite_img = None
        self.neurite_scale = None
        self.neurite_frangi = None
        self.composite_image = None
        self.neurons = None

        print('(A)\t Getting soma scale')
        self.get_soma_scale()

        print('(B)\t Getting soma seeds')
        self.get_soma_seeds()

        print('(C)\t Creating soma mask')
        self.get_soma_mask()

        print('(D)\t Creating neurite image')
        self.get_neurite_img()

        print('(E)\t Getting neurite scale')
        self.get_neurite_scale()

        print('(F)\t Getting filtered neurite image')
        self.frangi_filter()
        self.make_composite_image()

        print('(G)\t Getting neurons')
        self.get_neurons()

    def get_soma_scale(self):
        self.soma_scale = self.get_max_scale(self.binary)

    def get_soma_seeds(self):
        self.soma_seeds = self.get_seeds(self.img, self.soma_scale)

    def plot(self):
        fig = plt.figure(figsize=(10, 5), dpi=200)

        ax = fig.add_subplot(1, 2, 1)
        shape = self.img.GetSize()
        plt.title(f'{len(self.soma_seeds)} Unique seeds')
        soma_seeds = np.array(self.soma_seeds)
        plt.imshow(flatten(self.img), cmap='gray', interpolation='none', extent=[0, shape[0], 0, shape[1]])
        x = soma_seeds[:, 0]
        y = shape[0] - soma_seeds[:, 1]
        plt.plot(x, y, marker='x', markersize=20, linestyle='none')
        plt.axis('square')
        plt.axis('off')

        ax = fig.add_subplot(2, 5, 10)
        plt.title('Labeled neurons')
        plt.imshow(flatten(self.neurons), cmap='nipy_spectral', interpolation='none')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def plot_full_segmentation(self):
        assert self.save, f'Plotting is only possible if {type(self).__name__} was initialized with save=True'

        fig = plt.figure(figsize=(20, 7), dpi=200)

        ax = fig.add_subplot(2, 5, 1)
        plt.title(f'Original image, soma size {self.soma_scale} px')
        plt.imshow(flatten(self.img), cmap='gray', interpolation='none')
        plt.colorbar()
        plt.axis('off')

        ax = fig.add_subplot(2, 5, 2)
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

        ax = fig.add_subplot(2, 5, 3)
        plt.title('Soma mask')
        plt.imshow(flatten(self.soma_mask), cmap='gray', interpolation='none')
        plt.colorbar()
        plt.axis('off')

        ax = fig.add_subplot(2, 5, 4)
        plt.title(f'Neurites only, size={self.neurite_scale} px')
        plt.imshow(flatten(self.neurite_img), cmap='gray', interpolation='none')
        plt.colorbar()
        plt.axis('off')

        ax = fig.add_subplot(2, 5, 5)
        plt.title('Frangi')
        plt.imshow(flatten(self.neurite_frangi), cmap='gray', interpolation='none')
        plt.colorbar()
        plt.axis('off')

        ax = fig.add_subplot(2, 5, 6)
        plt.title('Composite image')
        plt.imshow(flatten(self.composite_image), cmap='gray', interpolation='none')
        plt.axis('off')
        plt.colorbar()

        ax = fig.add_subplot(2, 5, 7)
        plt.title('Original mask')
        plt.imshow(flatten(self.orig_mask), cmap='gray', interpolation='none')
        plt.colorbar()
        plt.axis('off')

        ax = fig.add_subplot(2, 5, 8)
        plt.title('Dilated mask')
        plt.imshow(flatten(self.dil_mask), cmap='gray', interpolation='none')
        plt.colorbar()
        plt.axis('off')

        ax = fig.add_subplot(2, 5, 10)
        plt.title('Labeled neurons')
        plt.imshow(flatten(self.neurons), cmap='nipy_spectral', interpolation='none')
        plt.colorbar()
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
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

    def get_seeds(self, img, scale):

        blobs = self.hessian_filter(img, [scale], dimension=0, scale=True)

        blobs = sitk.RescaleIntensity(blobs, 0, 65535) # 0-65535

        blobs = sitk.Cast(blobs, sitk.sitkUInt16)

        blobs = sitk.DiscreteGaussian(blobs, scale ** 2)  # Var is Sigma^2

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

        return self.prune_seeds(seeds, scale)

    @staticmethod
    def prune_seeds(seeds, radius):
        valid_seeds = copy.copy(seeds)
        init_seeds = len(valid_seeds)  # Initial number of seeds

        with tqdm(total=init_seeds) as pbar:
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
                        delta = distance(seed, compared_seed)
                        if delta < radius * 2:  # Scale is a radius
                            kill_indices.append(jj)
                valid_seeds = np.delete(valid_seeds, kill_indices, axis=0)

                ii += 1
                pbar.update(1)

            pbar.update(init_seeds - ii)  # Finish up loading bar

        return valid_seeds

    def get_soma_mask(self, print_time=False):

        mask = sitk.ConfidenceConnected(self.img, seedList=self.soma_seeds.tolist(),
                                        numberOfIterations=1,
                                        multiplier=2,
                                        initialNeighborhoodRadius=1,
                                        replaceValue=1)
        self.soma_mask = mask

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

        neurite_img = self.img * mask
        self.neurite_img = neurite_img

    def get_neurite_scale(self):
        binary, _ = apply_threshold(self.neurite_img, mthd='Max Entropy')

        neurite_scale = self.get_max_scale(binary)
        if neurite_scale < 1:
            neurite_scale = 1

        self.neurite_scale = neurite_scale

    def frangi_filter(self):
        if self.neurite_scale == 1:
            neurite_scale = 2
        else:
            neurite_scale = self.neurite_scale

        if neurite_scale < self.soma_scale / 2:
            scales = [neurite_scale]  # For when scale and soma scale too close together. np.arange will return []
        else:
            scales = np.arange(neurite_scale, self.soma_scale / 2).astype(int)

        frangi = self.hessian_filter(self.img, scales, dimension=1)

        frangi = sitk.RescaleIntensity(frangi, 0, 65535)  # 0-255

        frangi = sitk.Cast(frangi, sitk.sitkUInt16)

        self.neurite_frangi = frangi

    def hessian_filter(self, img, scales, dimension=0, scale=False):
        # TODO: Make this multi-threaded/parallelized

        frangi_filter = sitk.ObjectnessMeasureImageFilter()
        # frangi_filter.SetGamma(1)
        frangi_filter.SetAlpha(0.75)
        frangi_filter.SetBeta(0.75)
        frangi_filter.SetBrightObject(True)
        frangi_filter.SetScaleObjectnessMeasure(scale)
        frangi_filter.SetObjectDimension(dimension)

        gaussian_filter = sitk.DiscreteGaussianImageFilter()
        gaussian_filter.SetUseImageSpacing(False)

        images = []

        for ii, scale in tqdm(enumerate(scales)):
            gaussian_filter.SetVariance(int(scale ** 2))  # Sigma = Var^2
            img_blurred = gaussian_filter.Execute(img)
            img_blurred = sitk.Cast(img_blurred, sitk.sitkFloat32)

            frangi = frangi_filter.Execute(img_blurred)
            # frangi = sitk.Cast(frangi, self.PixelID)

            images.append(frangi)

        return self.max_between_stacks(images)

    @staticmethod
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

    def make_composite_image(self):
        mask = self.soma_mask

        mask = sitk.Cast(mask, sitk.sitkUInt16)

        label_image = sitk.ConnectedComponent(mask)
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.Execute(label_image, self.neurite_frangi)

        soma_sticker = sitk.Image(self.img.GetSize(), sitk.sitkUInt16)

        for label in stats.GetLabels():
            mean_I = stats.GetMean(label)
            soma_sticker += sitk.Cast((label_image == label), sitk.sitkUInt16) * mean_I

        comp = soma_sticker + self.neurite_frangi * sitk.Cast(mask == 0, sitk.sitkUInt16)
        # img_rescaled = sitk.RescaleIntensity(self.img)

        self.composite_image = comp

    def get_neurons(self):

        mask = sitk.ConfidenceConnected(self.composite_image, seedList=self.soma_seeds.tolist(),
                                        numberOfIterations=5,
                                        multiplier=1.75,
                                        initialNeighborhoodRadius=1,
                                        replaceValue=1)

        # _, threshold = apply_threshold(self.composite_image, mthd='Reyni Entropy')
        #
        # mask = sitk.ConnectedThreshold(self.composite_image,
        #                                seedList=self.soma_seeds.tolist(),
        #                                lower=threshold,
        #                                upper=255)

        print('\tFinished getting mask')

        # global_mask = self.max_between_stacks(mask, self.soma_mask)


        if self.save:
            self.orig_mask = mask

        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetKernelRadius(self.neurite_scale)
        dilate_filter.SetForegroundValue(1)

        mask = dilate_filter.Execute(mask)

        print('\tFinished dilation')

        if self.save:
            self.dil_mask = mask

        components = sitk.ConnectedComponent(mask)

        print('\tFinished getting components')

        self.neurons = components

    def __str__(self):
        return (f'Neuron(s) with \n\tSoma Scale = {self.soma_scale}\n'
                f'\tNeurite Scale = {self.neurite_scale}')
