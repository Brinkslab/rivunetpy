
import matplotlib.pyplot as plt
import SimpleITK as sitk

from rivuletpy import rtrace
from rivuletpy.swc import SWC
from rivuletpy.utils.io import loadswc
from rivuletpy.utils.plottools import flatten
from rivuletpy.utils.plottools import volume_show, volume_view

if __name__ == '__main__':

    original_filename = r'H:\rivuletpy\tests\data\test_swc_postprocessing\neuron_0001_original.r2t.swc'
    cleaned_filename = r'H:\rivuletpy\tests\data\test_swc_postprocessing\neuron_0001_cleaned.r2t.swc'
    soma_labeled_filename = r'H:\rivuletpy\tests\data\test_swc_postprocessing\neuron_0001.r2t.swc'


    fig, ax = plt.subplots(1, 3, dpi=300)

    swc_mat = loadswc(original_filename)
    swc = SWC()
    swc._data = swc_mat

    swc.as_image(ax=ax[0])
    ax[0].set_title('Before cleaning')
    print('.', end='')

    swc_mat = loadswc(cleaned_filename)
    swc = SWC()
    swc._data = swc_mat

    swc.as_image(ax=ax[1], fig=fig)
    ax[1].set_title('After cleaning')
    print('.', end='')

    swc_mat = loadswc(soma_labeled_filename)
    swc = SWC()
    swc._data = swc_mat

    swc.as_image(ax=ax[2], fig=fig)
    ax[2].set_title('Soma Labeled')
    print('.', end='')

    handles, labels = ax[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center')

    fig.show()

