from dataclasses import dataclass

from SimpleITK.SimpleITK import Image

from rivunetpy.swc import SWC

@dataclass
class Neuron:
    img: Image
    # soma_pos: tuple = None
    img_fname: str = None

    num: int = None

    swc: SWC = None
    swc_fname: str = None



    def add_SWC(self, swc):
        self.swc = swc
        self.img = None


