from rivunetpy.rivunetpy import Tracer

import time

if __name__ == '__main__':
    start_time = time.time()

    filename = r"H:\dataset_s0_c9_4D_20dB.tif"

    tracer = Tracer()
    tracer.set_file(filename)
    tracer.set_blur(6.5)
    tracer.set_output_dir(r'C:\Users\twhoekstra\Desktop\dataset_s0_s9')
    tracer.set_tolerance(0.05)
    neurons = tracer.execute()

    print(f'\n\nDone in {time.time() - start_time} s')
