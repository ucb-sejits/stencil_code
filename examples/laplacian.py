from stencil_code.stencil_kernel2 import Stencil
from ctree.util import Timer
import numpy
import numpy.testing

_ = numpy


import sys

import logging

logging.basicConfig(level=0)


class LaplacianKernel(Stencil):
    neighborhoods = [
        [(0, 0, 1), (0, 0, -1),
         (0, 1, 0), (0, -1, 0),
         (1, 0, 0), (-1, 0, 0)]
    ]

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(in_grid):
            out_grid[x] = -6 * in_grid[x]
            for y in self.neighbors(x, 0):
                out_grid[x] += in_grid[y]

    def kernel2(self, in_grid):
        print(in_grid[1, :10, :10])
        out_grid = numpy.empty_like(in_grid)
        for x in self.interior_points(in_grid):
            out_grid[x] = -6 * in_grid[x]
            for y in self.neighbors(x, 0):
                out_grid[x] += in_grid[y]
        print(out_grid[1, :10, :10])
        return out_grid

nx = 1024 if len(sys.argv) <= 1 else int(sys.argv[1])
ny = 1024 if len(sys.argv) <= 2 else int(sys.argv[2])
nz = 32 if len(sys.argv) <= 3 else int(sys.argv[3])

input_grid = numpy.random.random([nx, ny, nz]).astype(numpy.float32) * 1024

laplacian = LaplacianKernel(backend='ocl')
with Timer() as s_t:
    a = laplacian(input_grid)
print("Specialized time {:0.3f}s".format(s_t.interval))
print("a.shape {}".format(a.shape))

# too slow to compare directly, this will apply python method to just a subset
smaller_input_grid = input_grid[:64, :64, :nz]
py = LaplacianKernel(backend='python')
with Timer() as py_t:
    b = py(smaller_input_grid)
numpy.testing.assert_array_almost_equal(a[2:62, 2:62, 2:nz-2], b[2:62, 2:62, 2:nz-2], decimal=4)
print("PASSED")
print("Py time {:0.3f}".format(py_t.interval))
