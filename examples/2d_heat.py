from stencil_code.stencil_kernel import *
from stencil_code.stencil_grid import StencilGrid

import logging

logging.basicConfig(level=20)

import sys
import numpy
import math
import time
import random
from ctree.util import Timer

width = 256
height = 256
time_steps = 16


class Kernel(StencilKernel):
    neighbor_definition = [
        [(-1, 1, 0), (-1, -1, 0),
         (-1, 0, 1), (-1, 0, -1)],
        [(-1, 0, 0), (-1, 0, 0)]
    ]
    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            out_grid[x] = in_grid[x]
            for y in self.neighbors(x, 0):
                out_grid[x] += 0.125 * in_grid[y]
            for z in self.neighbors(x, 1):
                out_grid[x] -= 0.25 * in_grid[z]

kernel = Kernel(backend='ocl')
py_kernel = Kernel(backend='python')
in_grid = numpy.random.rand(time_steps, width, height).astype(numpy.float32) * 1024


with Timer() as t:
    a = kernel(in_grid)

with Timer() as py_time:
    b = py_kernel(in_grid)

# Python mode doesn't clamp yet
numpy.testing.assert_array_almost_equal(a[1:-1,1:-1,1:-1], b[1:-1,1:-1,1:-1])

print("Specialized Time: %.03fs" % t.interval)
print("Python Time: %.03fs" % py_time.interval)
