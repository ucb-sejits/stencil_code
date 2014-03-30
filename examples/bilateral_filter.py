from stencil_code.stencil_kernel import *
from stencil_code.stencil_grid import StencilGrid

import sys
import numpy
import math
import time

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


def distance(x, y):
    return math.sqrt(sum([(x[i]-y[i])**2 for i in range(0, len(x))]))

pixels = map(ord, list(image_in.read(width * height))) # Read in grayscale values
# pixels = image_in.read(width * height)    # Read in grayscale values
# intensity = float(sum(pixels))/len(pixels)

kernel = Kernel()
kernel.should_unroll = False
out_grid = StencilGrid([width, height])
out_grid.ghost_depth = radius
in_grid = StencilGrid([width, height])
in_grid.ghost_depth = radius
for x in range(-radius, radius+1):
    for y in range(-radius, radius+1):
        in_grid.neighbor_definition[1].append((x, y))

for x in range(0, width):
    for y in range(0, height):
        in_grid.data[(x, y)] = pixels[y * width + x]

gaussian1 = gaussian(stdev_d, radius*2)
gaussian2 = gaussian(stdev_s, 256)


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

kernel.kernel(in_grid, gaussian1, gaussian2, out_grid)


class Runner(object):
    def __call__(self, *args, **kwargs):
        # kernel = Kernel()
        # kernel.should_unroll = False
        out_grid = StencilGrid([width, height])
        out_grid.ghost_depth = radius
        in_grid = StencilGrid([width, height])
        in_grid.ghost_depth = radius
        for x in range(-radius, radius+1):
            for y in range(-radius, radius+1):
                in_grid.neighbor_definition[1].append((x, y))

        for x in range(0, width):
            for y in range(0, height):
                in_grid.data[(x, y)] = pixels[y * width + x]
        kernel.kernel(in_grid, gaussian1, gaussian2, out_grid)

import timeit
print("Average C version time: %.03fs" % timeit.timeit(stmt=Runner(),
      number=10))

exit()
numpy.set_printoptions(threshold=numpy.nan)

actual_grid = StencilGrid([width, height])
actual_grid.ghost_depth = radius
naive = Kernel()
naive.pure_python = True
with Timer() as t:
    naive.kernel(in_grid, gaussian1, gaussian2, actual_grid)
print("Python version time: %.03fs" % t.interval)

numpy.testing.assert_array_almost_equal(actual_grid.data,
                                        out_grid.data, decimal=5)

for x in range(0, width):
    for y in range(0,height):
        pixels[y * width + x] = out_grid.data[(x, y)]
out_intensity = float(sum(pixels))/len(pixels)
for i in range(0, len(pixels)):
    pixels[i] = min(255, max(0, int(pixels[i] * (intensity/out_intensity))))

image_out = open(sys.argv[4], 'wb')
image_out.write(''.join(map(chr, pixels)))
