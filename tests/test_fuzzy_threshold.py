from os import path
from rivunetpy.utils.io import *
from filtering.thresholding import fuzzy
import matplotlib.pyplot as plt

img = loadimg(path.join('tests', 'data', 'test.tif'))
thr = fuzzy(img, render=True)

