swc1_path = r'C:\Users\twh\PycharmProjects\rivuletpy\tests\data\Image5-no-soma.v3dpbd.swc'
swc2_path = r'C:\Users\twh\PycharmProjects\rivuletpy\tests\data\Image5-soma.v3dpbd.r2.swc'

import matplotlib.pyplot as plt

from rivuletpy.swc import SWC
from rivuletpy.utils.io import loadswc


swcs = []
for path in (swc1_path, swc2_path):
    swc_mat = loadswc(path)
    s = SWC()
    s._data = swc_mat
    swcs.append(s)

fig, ax = plt.subplots(1, len(swcs), dpi=300)
titles = ['No soma tracing', 'With soma tracing']
for ii, swc in enumerate(swcs):
    swc.as_image(ax=ax[ii])
    ax[ii].set_title(titles[ii])

fig.show()