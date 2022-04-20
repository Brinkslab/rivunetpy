import copy

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
        self.soma_scale = self.get_soma_scale()
        self.soma_seeds = self.get_soma_seeds()
        self.neurite_scale = self.get_neurite_scale()

    @staticmethod
    def get_max_scale(img, thresh='Otsu', fast=True):
        # This function is WAY too slow
        binary, _ = apply_threshold(img, mthd=thresh)

        if fast:
            distance_transform = sitk.SignedMaurerDistanceMapImageFilter()
            # distance_transform.SetUseImageSpacing(False)
            distance_transform.SetInsideIsPositive(True)
            # distance_transform.SetBackgroundValue(1)
            distance_transform.SetSquaredDistance(False)
            distance_img = distance_transform.Execute(binary)

            max_filter = sitk.MinimumMaximumImageFilter()
            _ = max_filter.Execute(distance_img)

            return int(max_filter.GetMaximum())
        else:
            erosions = 0

            eroded = copy.copy(binary)
            prev_eroded = eroded

            intensities = [get_intensity(eroded)]
            components = [-1]  # Dummy number that should never be physically possible

            # dilate_filter = sitk.BinaryDilateImageFilter()
            # dilate_filter.SetKernelRadius(1)
            # dilate_filter.SetForegroundValue(1)

            erode_filter = sitk.BinaryErodeImageFilter()
            erode_filter.SetKernelRadius(1)
            erode_filter.SetForegroundValue(1)

            while True:
                eroded = erode_filter.Execute(eroded)

                intensities.append(get_intensity(eroded))

                label_image = sitk.ConnectedComponent(eroded)
                stats = sitk.LabelIntensityStatisticsImageFilter()
                stats.Execute(label_image, img)
                components.append(len(stats.GetLabels()))

                if components[-1] == components[-2]:
                    break
                elif intensities[-1] == intensities[-2]:
                    break  # Back-up exit for when component finding fails

                erosions += 1

        return erosions

    def get_soma_scale(self):
        return self.get_max_scale(self.img, thresh='Otsu', fast=True)

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
        frangi = sitk.Cast(frangi, sitk.sitkUInt16)

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

    def get_rough_soma_cover(self):
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

        radius = self.soma_scale
        for seed in self.soma_seeds:
            if self.in_bounds(image, seed, radius):
                image[seed[0] - radius:seed[0] + radius, \
                seed[1] - radius:seed[1] + radius, \
                seed[2] - radius:seed[2] + radius] = 1

        x = sitk.GetArrayFromImage(image)
        image = image < 0.5
        return image

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
        print(small_box_corner_min, small_box_corner_max)
        print(big_box_corners)

        in_bounds_min = (big_box_corners[0, :] < small_box_corner_min).all()
        in_bounds_max = (big_box_corners[0, :] > small_box_corner_max).all()

        return in_bounds_min

    def get_neurite_scale(self):
        binary = apply_threshold(self.img, mthd='Otsu')
        mask = self.get_rough_soma_cover()
        plt.imshow(flatten(mask), cmap='gray')
        plt.colorbar()
        plt.show()

        return None

    def __str__(self):
        return (f'Neuron(s) with \n\tSoma Scale = {self.soma_scale}\n'
                f'\tNeurite Scale = {self.neurite_scale}')
