from rivunetpy import rtracenet

import time

if __name__ == '__main__':
    start_time = time.time()

    filename = r'data\dataset_s0_c9_4D.tif'
    results = rtracenet.trace_net(filename, asynchronous=True)

    print(f'\n\nDone in {time.time() - start_time} s')

