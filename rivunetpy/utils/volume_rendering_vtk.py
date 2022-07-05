# https://gist.github.com/somada141/b125fd74916018ffe028

# !/usr/bin/python
import io

import vtk
from PIL import Image
from scipy.stats.mstats import mquantiles
import SimpleITK as sitk
import vtk
import numpy as np
import matplotlib.pyplot as plt
from vtk.util.vtkConstants import VTK_UNSIGNED_CHAR

from rivunetpy.utils.color import RGB_from_hex

def get_tf(data): # Get transfer functions
    q = mquantiles(data.flatten(), [0.7, 0.98])
    q[0] = max(q[0], 1)
    q[1] = max(q[1], 1)
    tf = [[0, 0, 0, 0, 0], [q[0], 0, 0, 0, 0], [q[1], 1, 1, 1, 0.5], [data.max(), 1, 1, 1, 1]]
    return tf

def numpy2VTK(img, spacing=[1.0, 1.0, 1.0]):
    # evolved from code from Stou S.,
    # on http://www.siafoo.net/snippet/314
    importer = vtk.vtkImageImport()

    img_data = img.astype('uint8')
    img_string = img_data.tostring()  # type short
    dim = img.shape

    importer.CopyImportVoidPointer(img_string, len(img_string))
    importer.SetDataScalarType(VTK_UNSIGNED_CHAR)
    importer.SetNumberOfScalarComponents(1)

    extent = importer.GetDataExtent()
    importer.SetDataExtent(extent[0], extent[0] + dim[2] - 1,
                           extent[2], extent[2] + dim[1] - 1,
                           extent[4], extent[4] + dim[0] - 1)
    importer.SetWholeExtent(extent[0], extent[0] + dim[2] - 1,
                            extent[2], extent[2] + dim[1] - 1,
                            extent[4], extent[4] + dim[0] - 1)

    importer.SetDataSpacing(spacing[0], spacing[1], spacing[2])
    importer.SetDataOrigin(0, 0, 0)

    return importer


def default_vector_or_scale(input, default: tuple):
    output = None

    if input is None:  # Default vector
        output = default
    elif type(input) is tuple:  # Manual vector override
        output = input
    elif type(input) in (float, int):  # Scale default vector
        output = np.multiply(default, input)

    return output


def set_camera(ren, volume=None, pos=None, az=None, el=None, up=None, foc=None):
    # https://kitware.github.io/vtk-examples/site/Python/Medical/MedicalDemo4/
    camera = ren.GetActiveCamera()

    # First automatic camera setting. Manual overrides are allowed
    # Some vector inputs, e.g. up vector or pos vector also
    # accept a scalar input. This will scale the default vector
    if volume is not None:
        c = volume.GetCenter()
        DEFAULT_UP = (0, 0, -1)
        DEFAULT_POS = (c[0], c[1] - 400, c[2])
        DEFAULT_FOC = (c[0], c[1], c[2])

        up = default_vector_or_scale(up, DEFAULT_UP)
        pos = default_vector_or_scale(pos, DEFAULT_POS)
        foc = default_vector_or_scale(foc, DEFAULT_FOC)

        az = 30.0 if az is None else az
        el = 30.0 if el is None else el

    # Backup: Dumb defaults
    else:
        pos = (-256, -256, 512)
        az = 30.0
        el = 30.0
        up = (0, 0, -1)
        foc = (0, 0, 255.0)

    camera.SetViewUp(*up)
    camera.SetPosition(*pos)
    camera.SetFocalPoint(*foc)
    camera.Azimuth(az)
    camera.Elevation(el)
    camera.SetClippingRange(-1E6, 0)
    # camera.ResetCameraClippingRange()



def volumeRender(img, tf=[], spacing=[1.0, 1.0, 1.0], labeled=False,opacity=0.10, inverse=True):
    importer = numpy2VTK(img, spacing)

    # Transfer Functions
    opacity_tf = vtk.vtkPiecewiseFunction()
    color_tf = vtk.vtkColorTransferFunction()

    if len(tf) == 0:
        tf.append([img.min(), 0, 0, 0, 0])
        tf.append([img.max(), 1, 1, 1, 1])

    if labeled:
        labels = np.unique(img.flatten())
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ncc = len(colors)

        iicc = 0
        for ii, label in enumerate(labels):
            if label == 0:
                rgb = (0, 0, 0)
            else:
                hex = colors[iicc]
                rgb = RGB_from_hex(hex, norm=True)

                iicc = (iicc + 1) % ncc # Next color
            color_tf.AddRGBPoint(label, *rgb)
    else:
        for p in tf:
            if inverse:
                color_tf.AddRGBPoint(p[0], 1-p[1], 1-p[2], 1-p[3])
            else:
                color_tf.AddRGBPoint(p[0], p[1], p[2], p[3])

    for p in tf:
        opacity_tf.AddPoint(p[0], p[4]*opacity)

    # working on the GPU
    volMapper = vtk.vtkGPUVolumeRayCastMapper()
    volMapper.SetInputConnection(importer.GetOutputPort())

    # The property describes how the data will look
    volProperty = vtk.vtkVolumeProperty()
    volProperty.SetColor(color_tf)
    volProperty.SetScalarOpacity(opacity_tf)
    volProperty.ShadeOn()
    volProperty.SetInterpolationTypeToLinear()

    # working on the CPU
    # volMapper = vtk.vtkVolumeRayCastMapper()
    # compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
    # compositeFunction.SetCompositeMethodToInterpolateFirst()
    # volMapper.SetVolumeRayCastFunction(compositeFunction)
    # volMapper.SetInputConnection(importer.GetOutputPort())

    # The property describes how the data will look
    # volProperty = vtk.vtkVolumeProperty()
    # volProperty.SetColor(color_tf)
    # volProperty.SetScalarOpacity(opacity_tf)
    # volProperty.ShadeOn()
    # volProperty.SetInterpolationTypeToLinear()

    # Do the lines below speed things up?
    # pix_diag = 5.0
    # volMapper.SetSampleDistance(pix_diag / 5.0)
    # volProperty.SetScalarOpacityUnitDistance(pix_diag)

    vol = vtk.vtkVolume()
    vol.SetMapper(volMapper)
    vol.SetProperty(volProperty)

    return vol


def vtk_create_renderer(actors, light_follows=True):
    """
    Create a window, renderer, interactor, add the actors and start the thing

    Parameters
    ----------
    actors :  list of vtkActors

    Returns
    -------
    nothing
    """

    # create a rendering window and renderer
    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(1920, 1080)
    # ren.SetBackground( 1, 1, 1)

    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.SetLightFollowCamera(light_follows)

    for a in actors:
        # assign actor to the renderer
        ren.AddActor(a)

    return ren, renWin, iren


def vtk_basic(actors):
    """
    Create a window, renderer, interactor, add the actors and start the thing

    Parameters
    ----------
    actors :  list of vtkActors

    Returns
    -------
    nothing
    """

    # create a rendering window and renderer
    ren, renWin, iren = vtk_create_renderer(actors)

    # render
    renWin.Render()

    # enable user interface interactor
    iren.Initialize()
    iren.Start()





# https://nbviewer.org/urls/bitbucket.org/somada141/pyscience/raw/master/20141029_VolumeRendering/Material/VolumeRendering.ipynb
def vtk_show(renderer, width=400, height=300):
    """
    Takes vtkRenderer instance and returns an IPython Image with the rendering.
    """
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(width, height)
    renderWindow.Render()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetWriteToMemory(1)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    data = memoryview(writer.GetResult())

    return Image.open(io.BytesIO(data))


#####
