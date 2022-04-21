import numpy as np
import SimpleITK as sitk

from rivuletpy.swc import SWC
from rivuletpy.utils.volume_rendering_vtk import (volumeRender, vtk_create_renderer, set_camera,
                                                  vtk_show, vtk_basic, get_tf)


def flatten(image, whitebackground=False):

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
    return flat_image

def get_actors_from_args(args, labeled):
    actor_list = []

    num_images = 0
    for arg in args:
        if type(arg) is sitk.Image:
            num_images += 1
    if not len(labeled) == num_images:
        raise ValueError('Expected labeled keyword argument to contain an equal amount '
                        'True/Falses as there are input images.')

    for arg in args:
        if type(arg) is sitk.Image:
            img = arg
            data = sitk.GetArrayFromImage(img)
            # TODO: Need to flip either image or SWC. SWC is probably the wisest here.
            data = np.flip(data, axis=1)
            tf = get_tf(data)

            actor = volumeRender(data, tf=tf, spacing=img.GetSpacing(), labeled=labeled)
            actor_list = actor_list + actor

        if type(arg) is SWC:
            swc = arg
            actors = swc.swc_to_actors(offset=swc_Z_offset)
            actor_list = actor_list + actors

    return actor_list

def volume_view(*args, labeled=[]):

    actor_list = get_actors_from_args(args, labeled)

    vtk_basic(actor_list)


def volume_show(*args, labeled=[], w=400, h=300, pos=None, az=None, el=None, up=None, foc=None):
    actor_list = get_actors_from_args(args, labeled)

    ren, renWin, iren = vtk_create_renderer(actor_list, light_follows=False)

    set_camera(ren, volume=actor_list[0], pos=pos, az=az, el=el, up=up, foc=foc)

    img = vtk_show(ren)
    return img

def swc_view(swc):
    data = sitk.GetArrayFromImage(img)

    tf = get_tf(data)

    actor_list = volumeRender(data, tf=tf, spacing=img.GetSpacing(), labeled=labeled)

    vtk_basic(actor_list)
