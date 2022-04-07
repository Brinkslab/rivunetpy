
from rivuletpy.swc import SWC
from rivuletpy.utils.io import loadswc
from rivuletpy.utils.plottools import volume_show, volume_view, swc_view



if __name__ == '__main__':
    swc_mat = loadswc('data\Synthetic-no-bg.r2.swc')
    s = SWC()
    s._data = swc_mat

    volume_view(s)
    pass
