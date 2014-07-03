stencil_code -- A specializer built on the ASP SEJITS framework
-------

stencil_code is based on ctree
for more information on ctree see [ctree on github](http://github.com/ucb-sejits/ctree>).

[![Build Status](https://travis-ci.org/ucb-sejits/stencil_code.svg?branch=master)](https://travis-ci.org/ucb-sejits/stencil_code)
[![Coverage Status](https://coveralls.io/repos/ucb-sejits/stencil_code/badge.png?branch=master)](https://coveralls.io/r/ucb-sejits/stencil_code?branch=master)

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
Numpy convolve avg: 0.0357077
Specialized C with compile time avg: 0.1274197
Specialized C time avg without compile 0.0125766
Specialized OpenMP with compile time avg: 0.1360185
Specialized OpenMP time avg without compile 0.0123128
Specialized OpenCL with compile time avg: 0.0423939
Specialized OpenCL time avg without compile 0.0093015
```

Examples
=============
* [Simple](#simple)  
* [Bilateral Filter](#bilateralfilter)

<a name='simple'/>
### A simple kernel
```python
class Kernel(StencilKernel):
    def kernel(self, in_grid, out_grid):
        for x in out_grid.interior_points():
            for y in in_grid.neighbors(x, 1):
                out_grid[x] += in_grid[y]

kernel = Kernel()
width = 1024
in_grid = StencilGrid([width])
out_grid = StencilGrid([width])
for x in in_grid.interior_points():
    in_grid[x] = 1.0

kernel.kernel(in_grid, out_grid)
```

<a name='bilateralfilter'/>
### A bilateral filter
```python
width = int(sys.argv[2])
height = int(sys.argv[3])
image_in = open(sys.argv[1], 'rb')
stdev_d = 3
stdev_s = 70
radius = stdev_d * 3

class Kernel(StencilKernel):
    def kernel(self, in_img, filter_d, filter_s, out_img):
        for x in out_img.interior_points():
            for y in in_img.neighbors(x, 1):
                out_img[x] += in_img[y] * filter_d[int(distance(x, y))] *\
                    filter_s[abs(int(in_img[x] - in_img[y]))]


def gaussian(stdev, length):
    result = StencilGrid([length])
    scale = 1.0/(stdev*math.sqrt(2.0*math.pi))
    divisor = -1.0 / (2.0 * stdev * stdev)
    for x in range(length):
        result[x] = scale * math.exp(float(x) * float(x) * divisor)
    return result

# Instantiate a kernel
kernel = Kernel()

# Instantiate StencilGrids
out_grid = StencilGrid([width, height])
out_grid.ghost_depth = radius
in_grid = StencilGrid([width, height])
in_grid.ghost_depth = radius

# Define a neighborhood
for x in range(-radius, radius+1):
    for y in range(-radius, radius+1):
        in_grid.neighbor_definition[1].append((x, y))

# Copy image data into in_grid
for x in range(0, width):
    for y in range(0, height):
        in_grid.data[(x, y)] = pixels[y * width + x]

# Create our filters
gaussian1 = gaussian(stdev_d, radius*2)
gaussian2 = gaussian(stdev_s, 256)

kernel.kernel(in_grid, gaussian1, gaussian2, out_grid)
```
