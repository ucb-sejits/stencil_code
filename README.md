stencil_code -- A specializer built on the ASP SEJITS framework
-------

stencil_code is based on ctree
for more information on ctree see [ctree on github](http://github.com/ucb-sejits/ctree>).

[![Build Status](https://travis-ci.org/ucb-sejits/stencil_code.svg?branch=master)](https://travis-ci.org/ucb-sejits/stencil_code)
[![Coverage Status](https://coveralls.io/repos/ucb-sejits/stencil_code/badge.png?branch=master)](https://coveralls.io/r/ucb-sejits/stencil_code?branch=master)

Installation
============

```
pip install stencil_code --pre
```

Benchmarks Results
==================
Check out the benchmarks folders for some performance tests you can run 
on our own machine.  Here are the results on a MacBookPro 10,1 with the 
following specs.
* Processor  2.7 GHz Intel Core i7
* Memory  16 GB 1600 MHz DDR3
* Graphics  NVIDIA GeForce GT 650M 1024 MB

### benchmarks/convolve.py
```
Numpy convolve avg: 0.0370276
Specialized C with compile time avg: 0.1326465
Specialized C time avg without compile 0.0130333
Specialized OpenMP with compile time avg: 0.1278614
Specialized OpenMP time avg without compile 0.0125139
Specialized OpenCL with compile time avg: 0.0293867
Specialized OpenCL time avg without compile 0.0084686
```

![results_graph](https://raw.github.com/ucb-sejits/stencil_code/master/benchmarks/convolve_results.png)

Examples
=============
* [Simple](#simple)  
* [Bilateral Filter](#bilateralfilter)

<a name='simple'/>
### A simple kernel
```python
import numpy
from stencil_code.stencil_grid import StencilKernel

class SimpleKernel(StencilKernel):
    neighbor_definition = [[
        (-1, 1),  (0, 1),  (1, 1),
        (-1, 0),  (0, 0),  (1, 0),
        (-1, -1), (-1, 0), (-1, 1)
    ]]

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            for y in self.neighbors(x, 0):
                out_grid[x] += in_grid[y]


kernel = SimpleKernel()
width = 1024
in_grid = numpy.rand(width).astype(numpy.float32) * 1000

out_grid = kernel(in_grid)
```

<a name='bilateralfilter'/>
### A bilateral filter
```python
import numpy
from stencil_code.stencil_kernel import StencilKernel

width = int(sys.argv[2])
height = int(sys.argv[3])
image_in = open(sys.argv[1], 'rb')
stdev_d = 3
stdev_s = 70
radius = stdev_d * 3

class BilatKernel(StencilKernel):
    neighbor_definition = [[
        (-1, 1),  (0, 1),  (1, 1),
        (-1, 0),  (0, 0),  (1, 0),
        (-1, -1), (-1, 0), (-1, 1)
    ]]

    def kernel(self, in_img, filter_d, filter_s, out_img):
        for x in self.interior_points(out_img):
            for y in self.neighbors(x, 1):
                out_img[x] += in_img[y] * filter_d[
                    int(distance(x, y))] * \
                    filter_s[abs(int(in_img[x] - in_img[y]))]


def gaussian(stdev, length):
    result = StencilGrid([length])
    scale = 1.0/(stdev*math.sqrt(2.0*math.pi))
    divisor = -1.0 / (2.0 * stdev * stdev)
    for x in range(length):
        result[x] = scale * math.exp(float(x) * float(x) * divisor)
    return result

# Instantiate a kernel
kernel = BilatKernel()

# Get some input data
in_grid = numpy.random(width, height).astype(numpy.float32) * 255

# Create our filters
gaussian1 = gaussian(stdev_d, radius*2)
gaussian2 = gaussian(stdev_s, 256)

out_grid = kernel(in_grid, gaussian1, gaussian2)
```
