from stencil_code.stencil_kernel import *
from stencil_code.stencil_grid import StencilGrid

import logging

logging.basicConfig(level=20)

import sys
import numpy
import math
import time
import random

width = 130
height = 130
time_steps = 18


class Kernel(StencilKernel):
    def kernel(self, in_img, out_img):
        for x in out_img.interior_points():
            out_img[x] = in_img[x]
            for y in in_img.neighbors(x, 0):
                out_img[x] += 0.125 * in_img[y]
            for z in in_img.neighbors(x, 1):
                out_img[x] -= 0.125 * 2.0 * in_img[z]

kernel = Kernel(backend='ocl')
kernel.should_unroll = False
out_grid = StencilGrid([time_steps, width, height])
out_grid.ghost_depth = 1
in_grid = StencilGrid([time_steps, width, height])
in_grid.ghost_depth = 1

base = 1024
r = random.seed()
for i in range(width):
    for j in range(height):
        in_grid.data[(0, i, j)] = random.randrange(1024) * 1.0

in_grid.neighbor_definition[0] = [(-1, 1, 0), (-1, -1, 0),
                                  (-1, 0, 1), (-1, 0, -1)]
in_grid.neighbor_definition[1] = [(-1, 0, 0), (-1, 0, 0)]


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

with Timer() as t:
    kernel.kernel(in_grid, out_grid)
print("Time: %.03fs" % t.interval)
