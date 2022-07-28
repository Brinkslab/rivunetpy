from dataclasses import dataclass
import os

from SimpleITK.SimpleITK import Image
import numpy as np
import pandas as pd

from rivunetpy.swc import SWC

@dataclass
class Neuron:
    """Dataclass that stores information on a single neuron.

    Attributes:
        img (Image): Image of the neuron. Should not contain any other neurons.
        img_fname (str): Path pointing towards the image of the neuron.
        num (int): Number identifying the neuron.
        soma_radius: (int): Radius (scale) of the soma of the neuron in pixels.
        swc (rivunetpy.swc.SWC): Reconstructed neuron as SWC.
        swc_fname (str): Path pointing towards a copy of the reconstruction on
          disk.
        intensities (np.ndarray): Array containing the intensity trace. For
          voltage imaging, this is a proxy of the voltage trace.
        i_fname (str): Path pointing towards a copy of the intensity trace on
          disk.
    """
    img: Image

    img_fname: str = None

    num: int = None

    # soma_pos: tuple = None
    soma_radius: int = None

    swc: SWC = None
    swc_fname: str = None

    intensities: np.ndarray = None
    i_fname: str = None

    def add_SWC(self, swc):
        """Adds a reconstruction.

        Removes image to save on memory.
        """
        self.swc = swc
        self.img = None
