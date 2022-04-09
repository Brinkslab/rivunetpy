import numpy as np
import SimpleITK as sitk

from rivuletpy.swc import SWC
from rivuletpy.utils.volume_rendering_vtk import (volumeRender, vtk_create_renderer, set_camera,
                                                  vtk_show, vtk_basic, get_tf)


def imshow_flatten(ax, image: np.ndarray, **kwargs):
    if image.ndim == 3:  # Z stack
        z_axis_guess = np.argmin(image.shape)
        flat_image = np.max(image, axis=z_axis_guess)
    elif image.ndim == 2:
        flat_image = image
    else:
        raise TypeError('Expected an image with either 2 or 3 dimensions (Z Stack) '
                        f'but got an image of {image.ndim} dimensions of shape {image.shape}')
    ax.imshow(flat_image, **kwargs)


def volume_view(*args, labeled=False, swc_Z_offset=None):
    actor_list = []
    for arg in args:
        if type(arg) is sitk.Image:
            img = arg
            data = sitk.GetArrayFromImage(img)
            print(data.shape)
            data = np.flip(data, axis=1)
            tf = get_tf(data)

            actor = volumeRender(data, tf=tf, spacing=img.GetSpacing(), labeled=labeled)
            actor_list = actor_list + actor

        if type(arg) is SWC:
            swc = arg
            actors = swc.swc_to_actors(offset=swc_Z_offset)
            actor_list = actor_list + actors

    vtk_basic(actor_list)


def volume_show(img, labeled=False, w=400, h=300, pos=None, az=None, el=None, up=None, foc=None):
    data = sitk.GetArrayFromImage(img)  #

    tf = get_tf(data)

    actor_list = volumeRender(data, tf=tf, spacing=img.GetSpacing(), labeled=labeled)

    ren, renWin, iren = vtk_create_renderer(actor_list, light_follows=False)

    set_camera(ren, volume=actor_list[0], pos=pos, az=az, el=el, up=up, foc=foc)

    img = vtk_show(ren)
    return img

def swc_view(swc):
    data = sitk.GetArrayFromImage(img)

    tf = get_tf(data)

    actor_list = volumeRender(data, tf=tf, spacing=img.GetSpacing(), labeled=labeled)

    vtk_basic(actor_list)
