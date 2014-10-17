from stencil_code.stencil_kernel import *
from stencil_code.stencil_grid import StencilGrid
from ctree.util import Timer
import numpy


import sys

import logging

logging.basicConfig(level=0)
class LaplacianKernel(StencilKernel):
    neighbor_definition = [
        [(0, 0, 1), (0, 0, -1),
         (0, 1, 0), (0, -1, 0),
         (1, 0, 0), (-1, 0, 0)]
    ]

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(in_grid):
            out_grid[x] = -4 * in_grid[x]
            for y in self.neighbors(x, 0):
                out_grid[x] += in_grid[y]

# nx = int(sys.argv[1])
# ny = int(sys.argv[2])
# nz = int(sys.argv[3])
nx = 1024
ny = 1024
nz = 32

input_grid = numpy.random.rand(nx, ny, nz).astype(numpy.float32) * 1024

laplacian = LaplacianKernel(backend='ocl')
with Timer() as s_t:
    a = laplacian(input_grid)
print("Specialized time {:0.3f}s".format(s_t.interval))
## UNCOMMENT THIS TO RUN THE PYTHON VERSION, WARNING: VERY SLOW
# py = LaplacianKernel(backend='python')
# with Timer() as py_t:
#     b = py(input_grid)
# numpy.testing.assert_array_almost_equal(a, b)
# print("PASSED")
# print("Py time %0.3f", py_t)
