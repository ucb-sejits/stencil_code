"""
PDE heat flow simulation using a kernel.  Input is a 3d array where
each plane (each value of the first index) is a 2d
TODO: figure out if this works at all, seems like there is no guarantee
that the time steps will be run in the correct order
"""
from __future__ import print_function
from stencil_code.stencil_kernel import Stencil

import numpy
from ctree.util import Timer

width = 256
height = 256
time_steps = 16


class TwoDHeatFlow(Stencil):
    neighborhoods = [
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

if __name__ == '__main__':
    # import logging
    # logging.basicConfig(level=20)

    kernel = TwoDHeatFlow(backend='ocl')
    py_kernel = TwoDHeatFlow(backend='python')
    simulation_space = numpy.random.rand(time_steps, width, height).astype(numpy.float32) * 1024

    with Timer() as t:
        a = kernel(simulation_space)

    with Timer() as py_time:
        b = py_kernel(simulation_space)

    numpy.testing.assert_array_almost_equal(a, b)

    print("Specialized Time: %.03fs" % t.interval)
    print("Python Time: %.03fs" % py_time.interval)
