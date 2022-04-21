import copy
import time

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

    def __init__(self, img):
        self.img = img
        self.PixelID = self.img.GetPixelID()
        self.binary, self.threshold = apply_threshold(self.img, mthd='Otsu')
        self.soma_scale = self.get_soma_scale()
        self.soma_seeds = self.get_soma_seeds()
        self.neurite_scale = self.get_neurite_scale()
        self.get_global_mask()

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

    def get_soma_scale(self):
        return self.get_max_scale(self.binary)

    def get_soma_seeds(self):
        gaussian_filter = sitk.DiscreteGaussianImageFilter()
        gaussian_filter.SetUseImageSpacing(False)
        gaussian_filter.SetVariance(self.soma_scale ** 2)  # Sigma = Var^2
        img_blurred = gaussian_filter.Execute(self.img)
        img_blurred = sitk.Cast(img_blurred, sitk.sitkFloat32)

        frangi_filter = sitk.ObjectnessMeasureImageFilter()
        # frangi_filter.SetScaleObjectnessMeasure(...)
        frangi_filter.SetObjectDimension(0)
        frangi = frangi_filter.Execute(img_blurred)
        frangi = sitk.Cast(frangi, self.PixelID)

        blobs = frangi > 0

        label_image = sitk.ConnectedComponent(blobs)
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.Execute(label_image, self.img)

        soma_seeds = []
        for label in stats.GetLabels():
            cent = np.array(stats.GetCentroid(label)).astype(int).tolist()
            soma_seeds.append(tuple(cent))
        return self.prune_soma_seeds(soma_seeds)

    def prune_soma_seeds(self, seeds):
        valid_seeds = copy.copy(seeds)

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
                    if delta < self.soma_scale * 2:  # Soma scale is a radius
                        kill_indices.append(jj)
            valid_seeds = np.delete(valid_seeds, kill_indices, axis=0)

            ii += 1

        # self.plot_pruning(seeds, valid_seeds)
        return valid_seeds

    def plot_pruning(self, seeds, seeds_after):

        fig = plt.figure(figsize=(10, 3))

        ax = fig.add_subplot(1, 3, 1)
        plt.plot(np.array(seeds)[:, 0], np.array(seeds)[:, 1], 'ro')
        plt.axis('square')
        plt.xlim(0, self.img.GetSize()[1])
        plt.ylim(self.img.GetSize()[0], 0)
        plt.title(f'N={len(np.array(seeds)[:, 0])}')

        ax = fig.add_subplot(1, 3, 2)
        plt.plot(np.array(seeds_after)[:, 0], np.array(seeds_after)[:, 1], 'bo')
        plt.axis('square')
        plt.xlim(0, self.img.GetSize()[1])
        plt.ylim(self.img.GetSize()[0], 0)
        plt.title(f'N={len(np.array(seeds_after)[:, 0])}')

        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(flatten(self.img), cmap='gray')
        plt.colorbar()

        fig.show()

    def get_rough_soma_mask(self):
        # https://simpleitk.readthedocs.io/en/master/link_HelloWorld_docs.html
        pixelType = sitk.sitkUInt8
        imageSize = self.img.GetSize()
        image = sitk.Image(imageSize, pixelType)

        # Create blob images
        # cover_size = np.full(3, self.soma_scale).tolist()
        # for seed in self.soma_seeds[:2]:
        #     print(seed, cover_size)
        #     cover = sitk.GaussianSource(pixelType, imageSize, cover_size, seed)
        #     image = image + cover

        # Apply the blobs to the image
        # image, _ = apply_threshold(image, mthd='Otsu')
        print(len(self.soma_seeds))
        radius = self.soma_scale
        for seed in self.soma_seeds:
            if self.in_bounds(image, seed, radius):
                image[seed[0] - radius:seed[0] + radius, \
                seed[1] - radius:seed[1] + radius, \
                seed[2] - radius:seed[2] + radius] = 1
            else:
                print(f'Thrown out seed {seed}, too close to border.')

        return image

    def get_soma_mask(self, dilate=None, print_time=False):
        if print_time:
            start_time = time.time()

        mask = sitk.ConfidenceConnected(self.img, seedList=self.soma_seeds.tolist(),
                                        numberOfIterations=1,
                                        multiplier=1,
                                        initialNeighborhoodRadius=3,
                                        replaceValue=1)
        if dilate is not None:
            if type(dilate) not in (int, bool):
                raise ValueError('Expected dilate to be an integer representing the kernel' 
                                 f'Radius for the dilation filter, instead got {dilate}' 
                                 'Alternatively, pass True to automatically set size.')

            if type(dilate) is bool:
                dilate = self.soma_scale

            dilate_filter = sitk.BinaryDilateImageFilter()
            dilate_filter.SetKernelRadius(dilate)
            dilate_filter.SetForegroundValue(1)
            mask = dilate_filter.Execute(mask)

        if print_time:
            print(f'Got Soma Mask in {time.time() - start_time} s')
        return mask

    @staticmethod
    def in_bounds(big_box_corners, small_box_center, small_box_radius):
        if isinstance(big_box_corners, Image):
            big_box_corners = ((0, 0, 0), big_box_corners.GetSize())

        big_box_corners = np.array(big_box_corners)
        assert (big_box_corners[0, :] < big_box_corners[1, :]).all(), \
            'Bounding box corners should be formatted as: \n' \
            '[(x_min, y_min, z_min), (x_max, y_max, z_max)]'

        rr = small_box_radius
        small_box_corner_min = np.array([axval - rr for axval in small_box_center])
        small_box_corner_max = np.array([axval + rr for axval in small_box_center])

        in_bounds_min = (big_box_corners[0, :] < small_box_corner_min).all()
        in_bounds_max = (big_box_corners[0, :] > small_box_corner_max).all()

        return in_bounds_min

    def get_neurite_scale(self):
        mask = self.get_soma_mask(dilate=True, print_time=True)
        mask = mask == 0 # Invert mask
        mask = sitk.Cast(mask, self.img.GetPixelID())

        neurite_img = self.img * mask
        binary, _ = apply_threshold(neurite_img, mthd='Max Entropy')

        neurite_scale = self.get_max_scale(binary)
        if neurite_scale < 1:
            neurite_scale = 1

        return neurite_scale

    def frangi_filter(self):
        # TODO: Make this multi-threaded/parallelized
        # scales = [self.neurite_scale+1]
        scales = np.arange(self.neurite_scale+1, self.soma_scale/2).astype(int)
        print(f'Scales= {scales}')
        img_shape = self.img.GetSize()
        scales_stack_shape = (len(scales), img_shape[-1], *img_shape[:-1])

        scales_stack = np.zeros(scales_stack_shape)

        frangi_filter = sitk.ObjectnessMeasureImageFilter()
        # frangi_filter.SetGamma(1)
        frangi_filter.SetAlpha(0.75)
        frangi_filter.SetBeta(0.75)
        frangi_filter.SetBrightObject(True)
        frangi_filter.SetScaleObjectnessMeasure(False)
        frangi_filter.SetObjectDimension(1)

        gaussian_filter = sitk.DiscreteGaussianImageFilter()
        gaussian_filter.SetUseImageSpacing(False)

        for ii, scale in enumerate(scales):

            gaussian_filter.SetVariance(int(scale**2))  # Sigma = Var^2
            img_blurred = gaussian_filter.Execute(self.img)
            img_blurred = sitk.Cast(img_blurred, sitk.sitkFloat32)

            frangi = frangi_filter.Execute(img_blurred)
            # frangi = sitk.Cast(frangi, self.PixelID)

            frangi_np = sitk.GetArrayFromImage(frangi)

            scales_stack[ii] = frangi_np


        scales_stack_max = np.max(scales_stack, axis=0)
        scales_stack_max = sitk.GetImageFromArray(scales_stack_max)

        return scales_stack_max

    def get_global_mask(self):
        frangi = self.frangi_filter()

        # binary, _ = apply_threshold(frangi, mthd='Reyni Entropy')

        vs_binary, _ = apply_threshold(self.img, mthd='Otsu')

        mask = sitk.ConfidenceConnected(frangi, seedList=self.soma_seeds.tolist(),
                                        numberOfIterations=3,
                                        multiplier=1,
                                        initialNeighborhoodRadius=self.soma_scale,
                                        replaceValue=1)

        # label_seg = sitk.ConnectedComponent(seg)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 2, 1)
        plt.title('With filtering steps')
        plt.imshow(flatten(mask), cmap='gray', interpolation='none')
        plt.colorbar()

        ax = fig.add_subplot(1, 2, 2)
        plt.title('Simple threshold')
        plt.imshow(flatten(frangi), cmap='gray', interpolation='none')
        plt.colorbar()

        plt.tight_layout()
        plt.show()

    def __str__(self):
        return (f'Neuron(s) with \n\tSoma Scale = {self.soma_scale}\n'
                f'\tNeurite Scale = {self.neurite_scale}')
