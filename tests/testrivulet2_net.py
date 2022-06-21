from rivunetpy.rivunetpy import Tracer

import time

if __name__ == '__main__':
    start_time = time.time()

    filename = r'data\dataset_s0_c9_4D-20dB.tif'

    tracer = Tracer()
    tracer.set_file(filename)
    tracer.asynchronous_off()
    neurons = tracer.execute()

    print(f'\n\nDone in {time.time() - start_time} s')

