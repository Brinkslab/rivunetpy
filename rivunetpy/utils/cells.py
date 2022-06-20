from dataclasses import dataclass
import os

from SimpleITK.SimpleITK import Image
import numpy as np
import pandas as pd

from rivunetpy.swc import SWC

@dataclass
class Neuron:
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
        self.swc = swc
        self.img = None


