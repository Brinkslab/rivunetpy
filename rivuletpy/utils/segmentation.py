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
            self.orig_global_mask = None
            self.dil_global_mask = None

        print('(A)\t Getting soma scale')
        self.soma_scale = self.get_max_scale(self.binary)

        print('(B)\t Getting soma seeds')
        self.soma_seeds = self.get_seeds()

        print('(C)\t Creating soma mask')
        self.soma_mask = self.get_soma_mask()

        print('(D)\t Creating neurite image')
        self.neurite_img = self.get_neurite_img()

        print('(E)\t Getting neurite scale')
        self.neurite_scale = self.get_neurite_scale()

        print('(E)\t Getting filtered neurite image')
        self.neurite_frangi = self.hessian_filter()

        print('(G)\t Getting neurons')
        self.neurons = self.get_components()

    def plot_segmentation(self):
        assert self.save, f'Plotting is only possible if {type(self).__name__} was initialized with save=True'

        fig = plt.figure(figsize=(20, 10), dpi=200)

        ax = fig.add_subplot(2, 5, 1)
        plt.title('Original image')
        plt.imshow(flatten(self.img), cmap='gray', interpolation='none')
        # plt.colorbar()
        plt.axis('off')

        ax = fig.add_subplot(2, 5, 2)
        shape = self.img.GetSize()
        plt.title(f'{len(self.soma_seeds)} Unique seeds')
        soma_seeds = np.array(self.soma_seeds)
        plt.imshow(flatten(self.img), cmap='gray', interpolation='none', extent=[0, shape[0], 0, shape[1]])
        x = soma_seeds[:, 0]
        y = shape[0] - soma_seeds[:, 1]
        plt.plot(x, y, color='red', marker='x', markersize=20, linestyle='none')
        plt.axis('square')
        plt.axis('off')

        ax = fig.add_subplot(2, 5, 3)
        plt.title('Soma mask')
        plt.imshow(flatten(self.soma_mask), cmap='gray', interpolation='none')
        plt.axis('off')

        ax = fig.add_subplot(2, 5, 4)
        plt.title(f'Neurites only, size={self.neurite_scale} px')
        plt.imshow(flatten(self.neurite_img), cmap='gray', interpolation='none')
        plt.axis('off')


        ax = fig.add_subplot(2, 5, 5)
        plt.title('Frangi')
        plt.imshow(flatten(self.neurite_frangi), cmap='gray', interpolation='none')
        plt.axis('off')

        ax = fig.add_subplot(2, 5, 6)
        shape = self.img.GetSize()
        plt.title(f'{len(self.neurite_seeds)} Unique seeds')
        neurite_seeds = np.array(self.neurite_seeds)
        plt.imshow(flatten(self.img), cmap='gray', interpolation='none', extent=[0, shape[0], 0, shape[1]])
        x = neurite_seeds[:, 0]
        y = shape[0] - neurite_seeds[:, 1]
        plt.plot(x, y, color='blue', marker='x', markersize=20, linestyle='none')
        plt.axis('square')
        plt.axis('off')

        ax = fig.add_subplot(2, 5, 7)
        plt.title('Combined mask')
        plt.imshow(flatten(self.orig_global_mask), cmap='gray', interpolation='none')
        plt.axis('off')
        plt.colorbar()

        ax = fig.add_subplot(2, 5, 8)
        plt.title('Dilated mask')
        plt.imshow(flatten(self.dil_global_mask), cmap='gray', interpolation='none')
        plt.axis('off')

        ax = fig.add_subplot(2, 5, 10)
        plt.title('Labeled neurons')
        plt.imshow(flatten(self.neurons), cmap='nipy_spectral', interpolation='none')
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

    def get_seeds(self):

        blobs = self.hessian_filter(self.img, [self.soma_scale], dimension=0, scale=True)

        #blobs, _ = apply_threshold(blobs, mthd='Moments')

        plt.imshow(flatten(blobs), cmap='gray', interpolation='none')
        plt.colorbar()
        plt.show()

        blobs = sitk.Cast(blobs, self.PixelID)



        label_image = sitk.ConnectedComponent(blobs)
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.Execute(label_image, self.img)

        seeds = []
        for label in stats.GetLabels():
            cent = np.array(stats.GetCentroid(label)).astype(int).tolist()
            seeds.append(tuple(cent))

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # shape = self.img.GetSize()
        # plt.title(f'{len(self.soma_seeds)} Unique seeds')
        # soma_seeds = np.array(self.soma_seeds)
        # plt.imshow(flatten(self.img), cmap='gray', interpolation='none', extent=[0, shape[0], 0, shape[1]])
        # x = soma_seeds[:, 0]
        # y = shape[0] - soma_seeds[:, 1]
        # plt.plot(x, y, color='red', marker='x', markersize=20, linestyle='none')
        # plt.axis('square')
        # plt.axis('off')
        # plt.show()


        return self.prune_seeds(seeds, self.soma_scale)

    def get_seeds_old(self, img, scale, dimension=0):

        gaussian_filter = sitk.DiscreteGaussianImageFilter()
        gaussian_filter.SetUseImageSpacing(False)
        gaussian_filter.SetVariance(scale ** 2)  # Sigma = Var^2
        img_blurred = gaussian_filter.Execute(img)
        img_blurred = sitk.Cast(img_blurred, sitk.sitkFloat32)

        frangi_filter = sitk.ObjectnessMeasureImageFilter()
        # frangi_filter.SetScaleObjectnessMeasure(...)
        frangi_filter.SetObjectDimension(dimension)
        frangi = frangi_filter.Execute(img_blurred)
        frangi = sitk.Cast(frangi, self.PixelID)

        blobs = frangi > 0

        label_image = sitk.ConnectedComponent(blobs)
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.Execute(label_image, self.img)

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

            pbar.update(init_seeds-ii)  # Finish up loading bar

        return valid_seeds

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

    def get_soma_mask(self, dilate=True, print_time=False):
        if print_time:
            start_time = time.time()

        mask = sitk.ConfidenceConnected(self.img, seedList=self.soma_seeds.tolist(),
                                        numberOfIterations=1,
                                        multiplier=1,
                                        initialNeighborhoodRadius=1,
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

    def get_neurite_img(self):
        mask = self.soma_mask

        mask = mask == 0  # Invert mask
        mask = sitk.Cast(mask, self.img.GetPixelID())

        neurite_img = self.img * mask
        return neurite_img

    def get_neurite_scale(self):
        binary, _ = apply_threshold(self.neurite_img, mthd='Max Entropy')

        neurite_scale = self.get_max_scale(binary)
        if neurite_scale < 1:
            neurite_scale = 1

        return neurite_scale

    def frangi_filter(self):
        scales = np.arange(self.neurite_scale + 1, self.soma_scale / 2).astype(int)
        return self.hessian_filter(self.img, scales, dimension=1)

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
        if len(images) == 1 and type(images[0]) is list: # Allow for both list and multiple args input
            images = images[0]

        img_shapes = []
        img_types = []
        img: Image
        for img in images:
            assert type(img) is Image, ('Input image should by of the SimpleITK.SimpleITK.Image type, \n'
                                        f'instead is {type(img)}')
            img_shapes.append(img.GetSize())
            img_types.append(img.GetPixelID())

        assert len(set(img_shapes)) == 1, 'All the input images should have the same shape'
        assert len(set(img_types)) == 1, 'All the input images should have the same types'

        img_shape = img_shapes[0]
        stack_shape = (len(images), img_shape[-1], *img_shape[:-1])

        print(f'Single image has shape: {img_shape}, stack has shape: {stack_shape}')

        stack = np.zeros(stack_shape)

        for ii, img in enumerate(images):
            arr_buffer = sitk.GetArrayFromImage(img)
            stack[ii] = arr_buffer
            print(f'Added image {ii} to stack')

        stack_max = np.max(stack, axis=0)
        stack_max = sitk.GetImageFromArray(stack_max)

        stack_max = sitk.Cast(stack_max, int(img_types[0]))
        return stack_max

    def get_components(self):
        frangi = self.neurite_frangi

        if self.save:
            self.neurite_frangi = frangi

        mask = sitk.ConfidenceConnected(frangi, seedList=self.soma_seeds.tolist(),
                                        numberOfIterations=10,
                                        multiplier=1,
                                        initialNeighborhoodRadius=1,
                                        replaceValue=1)

        # threshold_filter = sitk.ConnectedThresholdImageFilter

        print('Finished getting mask')

        # global_mask = self.max_between_stacks(mask, self.soma_mask)
        global_mask = mask

        print('Finished getting maxima')

        if self.save:
            self.orig_global_mask = global_mask

        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetKernelRadius(self.neurite_scale)
        dilate_filter.SetForegroundValue(1)

        global_mask = dilate_filter.Execute(global_mask)

        print('Finished dilation')

        if self.save:
            self.dil_global_mask = global_mask

        components = sitk.ConnectedComponent(global_mask)

        print('Finished getting components')

        return components

    def __str__(self):
        return (f'Neuron(s) with \n\tSoma Scale = {self.soma_scale}\n'
                f'\tNeurite Scale = {self.neurite_scale}')
