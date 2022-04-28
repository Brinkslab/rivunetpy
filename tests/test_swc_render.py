
import matplotlib.pyplot as plt

from rivuletpy.swc import SWC
from rivuletpy.utils.io import loadswc
from rivuletpy.utils.plottools import volume_show, volume_view, swc_view



if __name__ == '__main__':
    swc_mat = loadswc('data\Image5.v3dpbd.r2.swc')
    s = SWC()
    s._data = swc_mat

    jj = 0
    for ii in range(1, 100, 5):
        s.set_view_density(ii)
        img = volume_show(s)

        plt.style.use('dark_background')
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(f'render_{str(jj).zfill(3)}.jpg')
        # plt.show()
        jj += 1
    pass
