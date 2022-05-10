import os

import time
from scipy.ndimage.interpolation import zoom
import SimpleITK as sitk

from rivuletpy.trace import R2Tracer
from rivuletpy.utils.io import loadimg, crop, swc2world, swc2vtk
from rivuletpy.utils.filtering import apply_threshold
from rivuletpy.trace import estimate_radius

# TODO: Remove this API

def show_logo():
    s = "====================Welcome to Rivulet2=================================="
    s += """\n\n8888888b.  d8b                   888          888           .d8888b.  
888   Y88b Y8P                   888          888          d88P  Y88b 
888    888                       888          888                 888 
888   d88P 888 888  888 888  888 888  .d88b.  888888            .d88P 
8888888P\"  888 888  888 888  888 888 d8P  Y8b 888           .od888P\"  
888 T88b   888 Y88  88P 888  888 888 88888888 888          d88P\"      
888  T88b  888  Y8bd8P  Y88b 888 888 Y8b.     Y88b.        888\"       
888   T88b 888   Y88P    \"Y88888 888  \"Y8888   \"Y888       888888888\n"""
    print(s)

def main(file=None, out=None, threshold=None, zoom_factor=1, save_soma=False,
         speed=False, quality=False, clean=True, non_stop=False, npush=0,
         silent=False, skeletonize=False, view=False, tracing_resolution=1.0, vtk=False, slicer=False):
    starttime = time.time()
    img = loadimg(file, tracing_resolution)
    img = sitk.GetArrayFromImage(img)

    imgshape = img.shape


    if threshold is None:
        bb, threshold = apply_threshold(img, mthd='Max Entropy')
    elif type(threshold) is str:
        bb, threshold = apply_threshold(img, mthd=threshold)

    # import matplotlib.pyplot as plt
    # from rivuletpy.utils.plottools import flatten
    # plt.imshow(flatten(bb))
    # plt.show()

    if not silent:
        pass
        # print('The shape of the image is', img.shape)
    # Modify the crop function so that it can crop somamask as well
    img, crop_region = crop(img, threshold)  # Crop by default

    if zoom_factor != 1.:
        if not silent:
            print('-- Zooming image to %.2f of original size' %
                  zoom_factor)
        img = zoom(img, zoom_factor)

    # Run rivulet2 for the first time
    tracer = R2Tracer(quality=quality,
                      silent=silent,
                      speed=speed,
                      clean=clean,
                      non_stop=non_stop,
                      skeletonize=skeletonize)

    swc, soma = tracer.trace(img, threshold)
    print('-- Finished: %.2f sec.' % (time.time() - starttime))

    # if skeletonized, re-estimate the radius for each node
    if skeletonize:
        print('Re-estimating radius...')
        swc_arr = swc.get_array()
        for i in range(swc_arr.shape[0]):
            swc_arr[i, 5] = estimate_radius(swc_arr[i, 2:5], img > threshold)
        swc._data = swc_arr

    if npush > 0:
        swc.push_nodes_with_binary(img > threshold, niter=npush)
    swc.reset(crop_region, zoom_factor)
    outpath = out if out else os.path.splitext(file)[
                                  0] + '.r2.swc'

    swc.save(outpath)

    # Save the soma mask if required
    if save_soma:
        soma.pad(crop_region, imgshape)
        soma.save((os.path.splitext(outpath)[0] + '.soma.tif'))

    # Save to vtk is required
    if vtk:
        print('Saving to SWC format...')
        swc.save(outpath.replace('.vtk', '.swc'))

        if not file.endswith('.tif'):
            print('Converting to world space...')
            img = sitk.ReadImage(file)
            swcarr = swc2world(swc.get_array(),
                               img,
                               [tracing_resolution] * 3,
                               slicer=slicer)
            swc._data[:, :7] = swcarr
        print('Saving to VTK format...')
        swc2vtk(swc, outpath.replace('.swc', '.vtk'))

    if view:
        from rivuletpy.utils.io import loadswc
        from rivuletpy.swc import SWC
        if os.path.exists(outpath):
            s = SWC()
            s._data = loadswc(outpath)
            s.view()
