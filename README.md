<!--
 Copyright (c) 2016, RivuletStudio, The University of Sydney, AU
 All rights reserved.

 This file is part of Rivuletpy <https://github.com/RivuletStudio/rivuletpy>

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
     3. Neither the name of the copyright holder nor the names of
        its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 -->

# RivuNetpy
## RivuNetpy is being developed to do parallel tracing of *networks* of neurons.
RivuNetpy can create reconstructions (structures of the neurons that can be understood and simulated by computers) from images containing multiple neurons. It can also analyze changes in intensity over time in an image. If the neuron was images using voltage imaging, these changes in intensity relate to the neuronal dynamics.
![alt text](meta/rivulet2_showcase.png "neuron showcase")

## RivuNetpy is a derivative of Rivuletpy
RivuNetpy is a Python 3 package for automatically reconstructing *multiple* neurons from 3D microscopic image stacks. RivuNetpy is a derivative of the [Rivuletpy](https://www.biorxiv.org/content/biorxiv/early/2017/11/27/109892.full.pdf) tracer.

RivuNetpy differs from Rivuletpy by:
* Tracing an image containing multiple neurons and creating a reconstruction for each individual neuron.
* Analyzing intensity changes in 3D stacks with a 4th time dimension.
* Saving reconstructions in a SWC format that can be directly imported into NetPyNE.
* Annotating sections in the reconstruction that belong to the soma.
* Having a Python interface rather than a Bash interface.
* RivuNetpy can read image metadata created in ImageJ and use this to scale reconstructions to um rather than pixel units.


## Installation

```
conda install -c brinkslab rivunetpy
```

## Usage

As an input image, RivuNetpy takes `.tif` files. The images should be 4D "hyperstacks" containing 3-dimensional images of the structure, with an additional dimension for the change in intensity over time. The ideal input is a hyperstack created in ImageJ. Rivunetpy can read the image metadata created in ImageJ to correctly scale the reconstructions. To summarize the restrictions on the input image:
1. It should be a 4D hyperstack (3D space, 1D time), saved in ImageJ
2. It should have distinguishable neurons
3. It should have metadata added to it before saving in ImageJ.

### Example 1
Reconstruct multiple neurons in a single image. Will export reconstructions to a folder named `hyperstack` in the same folder as the original image.

```python
from rivunetpy.rivunetpy import Tracer


tracer = Tracer()
tracer.set_file('hyperstack.tif')
tracer.execute()

```

### Example 2
Reconstruct multiple neurons in a single image. Apply a Gaussian blur preprocessing step with a kernel size of 3.

```python

tracer = Tracer()
tracer.set_file('hyperstack.tif')
tracer.set_blur(3)
tracer.execute()

```

### Example 3
Reconstruct multiple neurons in a single image. Explicitly define an output folder.

```python
tracer = Tracer()
tracer.set_file('hyperstack.tif')
tracer.set_output_dir(r'C:\Users\brinkslab\Desktop\My_Output_Folder')
tracer.execute()

```

### Example 4
Reconstruct multiple neurons in a single image. Catch the output reconstructions in Python. Reconstructions are contained in a list of `Neuron` objects. Each `Neuron` has a `swc` property containing the reconstructing as a tree structure.

```python

tracer = Tracer()
tracer.set_file('hyperstack.tif')
neurons = tracer.execute()

neuron = neurons[0] # Get first neuron
swc = neuron.swc
swc_matrix = swc._data
```

### Example 5
Reconstruct multiple neurons in a single image. By default, RivuNetpy, when run again on the same image, will reload the previous results from disk. This behavior can be turned off if undesirable.

```python

tracer = Tracer()
tracer.set_file('hyperstack.tif')
tracer.overwrite_cache_on()
neurons = tracer.execute()

```

## Development

### Install from source
Optionally you can install RivuNetpy from the source files

```
(riv)$ git clone https://github.com/brinkslab/rivunetpy
(riv)$ cd rivunetpy
(riv)$ python setup.py build
(riv)$ pip3 install .
```

### Todo
RivuNetpy is still in development. Some important facets that still need work can be divided into two categories. Firstly, RivuNetpy would be more useful for more people if it becomes a more general tool. specifically, it would be best to allow RivuNetpy to trace simple datasets created via a non-confocal, non-voltage imaging setup, e.g. 2D structural data. To do this, three changes are needed.
* EASY: Removing the requirement for having a temporal aspect to the data, and when this is the case forgoing the intensity recording step. The API should be extended such that the user can turn on this "structure-only" mode.]
* MEDIUM: Generalize the segmentation algorithm used in RivuNetpy. Ideally, integrating a segmentation method that is more intelligent (AI).
* HARD: Allowing for 2-dimensional structural data rather than requiring 3-dimensional structural data. Rivuletpy needs a 3D image to perform a reconstruction. A change in the source code is needed to allow it to accurately trace 2D images.

For the futher development for use in analyzing voltage imaging data, a couple big changes are needed.
* EASY: Selecting which point to retrieve voltage imaging data from (currently hard-coded to the soma). Maybe using a GUI of some sorts.
* MEDIUM: Use the voltage dynamics during the segmentation step. Differences in activity can be used to discriminate between two neurons, rather that having to rely soley on structural information. This could be allow for the segmentation algorithm to discern between two overlapping neurons in 2D images.
* HARD: Reconstructing and labeling axons. Currently, RivuNetpy considers all neurites to be dendrites. Voltage imaging infromation could be used to correctly label axons.
* HARD: Reconstructing synaptic connections between neurons from voltage imaging data.
