from rivunetpy import rtracenet

import time

if __name__ == '__main__':
    start_time = time.time()

    filename = r'data\MAX_dataset_s0_c9-SNR_25.tif'
    rtracenet.trace_net(filename)

    print(f'\n\nDone in {time.time() - start_time} s')

