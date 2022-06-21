import os

from rivunetpy.swc import clean
from rivunetpy.utils.extensions import RIVULET_2_TREE_SWC_EXT

if __name__ == '__main__':

    folder = 'data\ground_truths'

    old_swcs = []
    for file in os.listdir(folder):
        if os.path.splitext(file)[-1] == '.swc' and RIVULET_2_TREE_SWC_EXT not in file:
            old_swcs.append(os.path.join(folder, file))
    print(old_swcs)
    clean(old_swcs)