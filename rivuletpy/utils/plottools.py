import numpy as np
import SimpleITK as sitk

from rivuletpy.utils.volume_rendering_vtk import (volumeRender, vtk_create_renderer, set_camera,
                                                  vtk_show, vtk_basic, get_tf)


def imshow_flatten(ax, image: np.ndarray, **kwargs):
    if image.ndim == 3:  # Z stack
        flat_image = np.max(image, axis=0)
    elif image.ndim == 2:
        flat_image = image
    else:
        raise TypeError('Expected an image with either 2 or 3 dimensions (Z Stack) '
                        f'but got an image of {image.ndim} dimensions of shape {image.shape}')
    ax.imshow(flat_image, **kwargs)


def volume_view(img, labeled=False):
    data = sitk.GetArrayFromImage(img)

    tf = get_tf(data)

    actor_list = volumeRender(data, tf=tf, spacing=img.GetSpacing(), labeled=labeled)

    vtk_basic(actor_list)


def volume_show(img, labeled=False, w=400, h=300, pos=None, az=None, el=None, up=None, foc=None):
    data = sitk.GetArrayFromImage(img)  #

    tf = get_tf(data)

    actor_list = volumeRender(data, tf=tf, spacing=img.GetSpacing(), labeled=labeled)

    ren, renWin, iren = vtk_create_renderer(actor_list, light_follows=False)

    set_camera(ren, volume=actor_list[0], pos=pos, az=az, el=el, up=up, foc=foc)

    img = vtk_show(ren)
    return img
