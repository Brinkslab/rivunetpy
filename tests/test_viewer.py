from rivuletpy.utils.io import loadswc
from rivuletpy.swc import SWC

swc_mat = loadswc('data/Series021.v3dpbd.swc')
s = SWC()
s._data = swc_mat
s.view()
input("Press any key to continue...")