from stencil_code.stencil_kernel2 import Stencil
from stencil_code.stencil_grid import StencilGrid
from ctree.util import Timer
import numpy


import sys

import logging

logging.basicConfig(level=0)


class LaplacianKernel(Stencil):
    Stencil.set_neighbor_definition([
        [(0, 0, 1), (0, 0, -1),
         (0, 1, 0), (0, -1, 0),
         (1, 0, 0), (-1, 0, 0)]
    ])

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(in_grid):
            out_grid[x] = -4 * in_grid[x]
            for y in self.neighbors(x, 0):
                out_grid[x] += in_grid[y]

nx = 1024 if len(sys.argv) <= 1 else int(sys.argv[1])
ny = 1024 if len(sys.argv) <= 2 else int(sys.argv[2])
nz = 32  if len(sys.argv) <= 3 else int(sys.argv[3])

input_grid = numpy.random.rand(nx, ny, nz).astype(numpy.float32) * 1024

laplacian = LaplacianKernel(backend='ocl')
with Timer() as s_t:
    a = laplacian(input_grid)
print("Specialized time {:0.3f}s".format(s_t.interval))


## UNCOMMENT THIS TO RUN THE PYTHON VERSION, WARNING: VERY SLOW
py = LaplacianKernel(backend='python')
with Timer() as py_t:
    b = py(input_grid)
numpy.testing.assert_array_almost_equal(a, b)
print("PASSED")
print("Py time %0.3f", py_t)
