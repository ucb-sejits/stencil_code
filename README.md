stencil_code -- A specializer built on the CTree SEJITS framework
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
from stencil_code.stencil_kernel import Stencil

class SimpleKernel(Stencil):
    neighborhoods = [[
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
from stencil_code.neighborhood import Neighborhood
from stencil_code.stencil_kernel import Stencil
import math

# import logging
# logging.basicConfig(level=20)


class BilateralFilter(Stencil):
    def __init__(self, radius=3, backend='ocl'):
        super(BilateralFilter, self).__init__(
            neighborhoods=[Neighborhood.moore_neighborhood(radius=radius, dim=2)],
            backend=backend,
            should_unroll=False
        )

    def distance(self, x, y):
        return math.sqrt(sum([(x[i]-y[i])**2 for i in range(0, len(x))]))

    def kernel(self, in_img, filter_d, filter_s, out_img):
        for i in self.interior_points(out_img):
            for j in self.neighbors(i, 0):
                out_img[i] += in_img[j] * filter_d[int(self.distance(i, j))] *\
                    filter_s[abs(int(in_img[i] - in_img[j]))]


def gaussian(stdev, length):
    result = numpy.zeros(length).astype(numpy.float32)
    scale = 1.0/(stdev*math.sqrt(2.0*math.pi))
    divisor = -1.0 / (2.0 * stdev * stdev)
    for x in xrange(length):
        result[x] = scale * math.exp(float(x) * float(x) * divisor)
    return result


if __name__ == '__main__':
    import sys

    width = int(sys.argv[2])
    height = int(sys.argv[3])
    image_in = open(sys.argv[1], 'rb')
    out_filename = "/dev/null" if len(sys.argv) < 5 else sys.argv[4]
    stdev_d = 3
    stdev_s = 70
    radius = stdev_d * 3

    pixels = map(ord, list(image_in.read(width * height))) # Read in grayscale values
    intensity = float(sum(pixels))/len(pixels)
    print("intensity {}".format(intensity))

    ocl_bilateral_filter = BilateralFilter(radius, backend='ocl')

    # convert input stream into 2d array
    in_grid = numpy.zeros([height, width], numpy.float32)
    for i in xrange(height):
        for j in xrange(width):
            in_grid[i, j] = pixels[i * width + j]

    gaussian1 = gaussian(stdev_d, radius*2)
    gaussian2 = gaussian(stdev_s, 256)

    ocl_out_grid = ocl_bilateral_filter(in_grid, gaussian1, gaussian2)

    for i in xrange(height):
        for j in xrange(width):
            pixels[i * width + j] = (ocl_out_grid[i, j])

    # print(pixels)
    print("sum pix sum {} len {}".format(sum(pixels), len(pixels)))
    out_intensity = float(sum(pixels))/len(pixels)
    for i in range(0, len(pixels)):
        pixels[i] = min(255, max(0, int(pixels[i] * (intensity/out_intensity))))

    image_out = open(out_filename, 'wb')
    image_out.write(''.join(map(chr, pixels)))

```
