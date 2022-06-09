from itertools import cycle

import numpy as np
import SimpleITK as sitk
import matplotlib as plt

from PIL import Image
from matplotlib.axes import Axes
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from rivunetpy.swc import SWC
from rivunetpy.utils.cells import Neuron
from rivunetpy.utils.volume_rendering_vtk import (volumeRender, vtk_create_renderer, set_camera,
                                                  vtk_show, vtk_basic, get_tf)

def flatten(image, as_sitk=False, whitebackground=False):

    if type(image) is sitk.Image:
        image = sitk.GetArrayFromImage(image)

    if image.ndim == 3:  # Z stack
        z_axis_guess = np.argmin(image.shape)
        if not whitebackground:
            flat_image = np.max(image, axis=z_axis_guess)
        else:
            flat_image = np.min(image, axis=z_axis_guess)
    elif image.ndim == 2:
        flat_image = image
    else:
        raise TypeError('Expected an image with either 2 or 3 dimensions (Z Stack) '
                        f'but got an image of {image.ndim} dimensions of shape {image.shape}')

    if as_sitk:
        return sitk.GetImageFromArray(flat_image)
    else:
        return flat_image


def _plot_swc(swc: SWC, ax, center_fig, c=None, linewidths=None, unitstring=''):
    MAX_SEGMENTS = 2_500

    smallest_dim = 2 # Assume confocal stack is thinnest in Z

    segments = swc._data.shape[0]
    if segments > MAX_SEGMENTS:
        print(f'Too many ({segments}) segments to plot. Limiting plot to {MAX_SEGMENTS} segments.')
        segments = MAX_SEGMENTS

    XYZ_indecies = np.array([0, 1, 2])
    #ax.set_ylim(ax.get_ylim()[::-1])


    # Compute the center of mass
    center = swc._data[:, 2:5].mean(axis=0)

    translated = swc._data[:, 2:5] - \
                 np.tile(center, (swc._data.shape[0], 1)) * center_fig

    lid = swc._data[:, 0]

    if linewidths is None:
        linewidths = np.full(segments, 1.5)

    for ii in range(segments):

        TypeID = swc._data[ii, 1]
        if c is None:
            color = swc.get_TypeID_color(TypeID)
            # label = swc.get_TypeID_label(TypeID)
        else:
            color = c

        # Draw a line between this node and its parent
        if ii < swc._data.shape[0] - 1 and swc._data[ii, 6] == swc._data[ii - 1, 0]:
            # Fast track: if ParentID of current node matches SampleID of previous node
            coord_index = np.delete(XYZ_indecies, smallest_dim)

            ax.plot(
                translated[ii - 1:ii + 1, coord_index[0]],
                translated[ii - 1:ii + 1, coord_index[1]],
                color=color,
                linewidth=linewidths[ii]
            )

        else:
            # If there is a "less nice" data structure (i.e. parent node NOT before current node in data)
            pid = swc._data[ii, 6]
            pidx = np.argwhere(pid == lid)      # Find all parentIDs
            pidx = np.squeeze(pidx, axis=1)     # Remove unnecessary dimension
            for pidx_sel in pidx:
                indices = np.concatenate([[ii], [pidx_sel]])
                coord_index = np.delete(XYZ_indecies, smallest_dim)
                ax.plot(
                    translated[indices, coord_index[0]],
                    translated[indices, coord_index[1]],
                    color=color,
                    linewidth=linewidths[ii]
                )

def plot_segmentation(neurons, ax=None):

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])

    neuron_fnames = [neuron.img_fname for neuron in neurons]

    for ii, fname in enumerate(neuron_fnames):
        img = sitk.ReadImage(fname)

        shape = img.GetSize()

        c = next(colors)
        R, G, B = matplotlib.colors.to_rgb(c)

        N = 256
        aa_cmap = np.ones((N, 4))
        aa_cmap[:, 0] = np.linspace(0, R, N)  # R
        aa_cmap[:, 1] = np.linspace(0, G, N)  # G
        aa_cmap[:, 2] = np.linspace(0, B, N)  # B

        ######### Create color map with step in alpha channel ############
        # Convert threshold to value between 0-1
        if img.GetPixelID() == sitk.sitkUInt16:
            max_px = 2 ** 16 - 1
        else:
            max_px = 2 ** 8 - 1
        step = 1 / max_px
        aa_cmap[:, -1] = np.linspace(0, 1, N) > step
        ###################################################################

        aa_cmap = ListedColormap(aa_cmap)

        ax.imshow(flatten(img),
                     cmap=aa_cmap,
                     # extent=[0, shape[0], 0, shape[1]],
                     alpha=1)



def plot_swcs(swcs, **kwargs):
    """Exports SWC data as 2D image.

    When a `matplotlib.axes.Axes` object is added as a keyword argument `ax`, the 2D data will be
    added directly to the plot. Alternatively, no keyword arguments, or keyword arguments specifying
    a `matplotlib.figure.Figure` object (`figsize`, `frameon`, etc.) can be entered, resulting in
    the function returning a 2D image of the SWC.

    Args:
        **kwargs: Keyword argument `ax` (matplotlib.axes.Axes) OR keyword arguments from `matplotlib.figure.Figure`

    Returns:
        PIL.Image.Image: When set to output an image

    Examples:

        Setup:\n
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(1, 1)
        >>> swc_mat = loadswc(example.swc)
        >>> s = SWC()
        >>> s._data = swc_mat

        As image:\n
        >>> im = s.as_image()
        >>> ax.imshow(im, cmap='gray')

        Direct to plot:\n
        >>> s.as_image(ax=ax)

    """


    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])

    if isinstance(swcs, (list, tuple)):
        pass
    elif isinstance(swcs, SWC):
        swcs = [swcs]
    else:
        raise ValueError(f'Input of type {type(swcs)}, should either be a \n '
                         'rivunetpy.swc.SWC object or a list of SWC objects')

    if 'center_fig' in kwargs:
        center_fig = bool(kwargs['center_fig'])
    else:
        center_fig = False

    # Plot square by default, always turn off frame
    if kwargs == {}:
        kwargs = {'figsize': [6.4, 6.4]}
        kwargs['frameon'] = False

    if 'ax' in kwargs:  # Plotting directly to axes
        AX_SET = True
        ax = kwargs['ax']
        ax.set_aspect('equal', adjustable='box')
    else:
        AX_SET = False
        fig = Figure(**kwargs)

        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot()

    if 'fig' in kwargs: # Line thickness can be varied thanks to DPI info

        fig = kwargs['fig']

        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig_width, height = bbox.width, bbox.height

        fig_width *= 72 # Units pt

        vary_line = True
    else:
        vary_line = False

    if 'units' in kwargs:
        unitstring = f' [{kwargs["units"]}]'
    else:
        unitstring = ''

    ax.set_xlabel(f'X{unitstring}')
    ax.set_ylabel(f'Y{unitstring}')
    ax.invert_yaxis()

    # First iteration
    for swc in swcs:
        _plot_swc(swc, ax, center_fig, c=next(colors))

    # Optional second iteration for setting line thickness
    if vary_line:
        # Use existing plot to retrieve width of plot in px. Needed to accurately scale linewidths
        limits = ax.get_xlim()
        fig_px_per_img_px = fig_width / abs(limits[1] - limits[0])
        ax.cla()  # Clear old plot

        for swc in swcs:
            linewidths = swc._data[:, 5] * 2   # Linewidths in data units
            linewidths *= fig_px_per_img_px     # Linewidths in pt

            _plot_swc(swc, ax, center_fig, c=next(colors), linewidths=linewidths)

    if AX_SET:
        return None
    else:
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        im = Image.fromarray(rgba)

        return im


def get_actors_from_args(args, labeled=False):
    actor_list = []

    num_images = 0
    for arg in args:
        if type(arg) is sitk.Image:
            num_images += 1

    if labeled and not len(labeled) == num_images:
        raise ValueError('Expected labeled keyword argument to contain an equal amount '
                        'True/Falses as there are input images.')

    # Colors for coloring SWCs
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])
    color = next(colors)

    for arg in args:
        if type(arg) is sitk.Image:
            img = arg
            data = sitk.GetArrayFromImage(img)
            # TODO: Need to flip either image or SWC. SWC is probably the wisest here.
            #data = np.flip(data, axis=1)
            data = np.transpose(data, axes=[2, 1, 0])
            tf = get_tf(data)

            actor = volumeRender(data, tf=tf, spacing=img.GetSpacing(), labeled=labeled)

        if type(arg) is SWC:
            swc = arg
            actor = swc.as_actor(color=color, centered=False)
            color = next(colors)

        actor_list.append(actor)

    return actor_list

def volume_view(*args, labeled=False):

    actor_list = get_actors_from_args(args, labeled=labeled)

    vtk_basic(actor_list)


def volume_show(*args, labeled=[], w=400, h=300, pos=None, az=None, el=None, up=None, foc=None):
    actor_list = get_actors_from_args(args, labeled)

    ren, renWin, iren = vtk_create_renderer(actor_list, light_follows=False)

    set_camera(ren, volume=actor_list[0], pos=pos, az=az, el=el, up=up, foc=foc)

    img = vtk_show(ren)
    return img

# def swc_view(swc):
#     data = sitk.GetArrayFromImage(img)
#
#     tf = get_tf(data)
#
#     actor_list = volumeRender(data, tf=tf, spacing=img.GetSpacing(), labeled=labeled)
#
#     vtk_basic(actor_list)
