from rivunetpy.rivunetpy import Tracer

import time

if __name__ == '__main__':
    start_time = time.time()

    # filename = r'H:\Duet\Visualizations\MicroscopeImages\HyperStack.tif'
    filename = r"C:\Users\twh\Desktop\HyperStack.tif"

    tracer = Tracer()
    tracer.set_file(filename)
    tracer.asynchronous_off()
    tracer.set_tolerance(0.15)
    neurons = tracer.execute()

    print(f'\n\nDone in {time.time() - start_time} s')

