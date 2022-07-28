import os
"""Extensions used for RivuNetpy.

By default, RivuNetpy uses .rnp.tif for image files and .rnp.swc for
reconstructions.
"""

RIVULET_2_TREE_IMG_EXT = '{}rnp{}tif'.format(os.extsep, os.extsep)
RIVULET_2_TREE_SWC_EXT = '{}rnp{}swc'.format(os.extsep, os.extsep)
