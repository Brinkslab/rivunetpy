
import matplotlib.pyplot as plt

from rivuletpy import rtrace
from rivuletpy.swc import SWC
from rivuletpy.utils.io import loadswc
from rivuletpy.utils.plottools import flatten
from rivuletpy.utils.plottools import volume_show, volume_view

if __name__ == '__main__':

    original_filename = r'H:\rivuletpy\tests\data\synthetic-3-cells\neuron_0001.r2t.swc'
    cleaned_filename = r'H:\rivuletpy\tests\data\synthetic-3-cells\neuron_0001_clean.r2t.swc'

    fig, ax = plt.subplots(1, 2)

    swc_mat = loadswc(original_filename)
    swc = SWC()
    swc._data = swc_mat

    swc.as_image(ax=ax[0])
    ax[0].set_title('Before cleaning')

    # print('Before cleaning')
    # print(swc._data[:5,:])

    swc.clean()

    swc.as_image(ax=ax[1])
    ax[1].set_title('After cleaning')

    # print('After cleaning')
    # print(swc._data[:5, :])

    swc.save(cleaned_filename)

    handles, labels = ax[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center')



    fig.show()

    from rivuletpy import compareswc
