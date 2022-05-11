from rivuletpy import rtrace_net

import SimpleITK as sitk

if __name__ == '__main__':
    # 213.82 S for cell 4, aka neuron_0000
    filename = r'data\synthetic-3-cells.tif'
    rtrace_net.trace_net(filename, speed=True, asynchronous=False)

