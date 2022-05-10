from rivuletpy import rtrace_net

import SimpleITK as sitk

if __name__ == '__main__':

    rtrace_net.trace_net('data\synthetic-3-cells.tif', speed=True)

        # from rivuletpy.utils.plottools import flatten
        # import matplotlib.pyplot as plt
        # plt.imshow(flatten(self._bimg))
        # plt.show()