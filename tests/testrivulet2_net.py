from rivuletpy import rtrace_net

import SimpleITK as sitk
import time
if __name__ == '__main__':
    start_time = time.time()

    filename = r'data\synthetic-3-cells.tif'
    rtrace_net.trace_net(filename)

    print(f'\n\nDone in {time.time() - start_time} s')

