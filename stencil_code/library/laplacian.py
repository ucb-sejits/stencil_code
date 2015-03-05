from __future__ import print_function

from stencil_code.stencil_kernel import Stencil, product
from ctree.util import Timer
import numpy
import numpy.testing

import sys


class LaplacianKernel(Stencil):
    neighborhoods = [
        [(0, 0, 1), (0, 0, -1),
         (0, 1, 0), (0, -1, 0),
         (1, 0, 0), (-1, 0, 0)]
    ]

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(in_grid):
            out_grid[x] = -4 * in_grid[x]
            for y in self.neighbors(x, 0):
                out_grid[x] += in_grid[y]


if __name__ == '__main__':  # pragma: no cover
    import logging
    logging.basicConfig(level=0)

    nx = 8 if len(sys.argv) <= 1 else int(sys.argv[1])
    ny = 32 if len(sys.argv) <= 2 else int(sys.argv[2])
    nz = 128 if len(sys.argv) <= 3 else int(sys.argv[3])

    numpy.random.seed(0)
    input_grid = numpy.random.random([nx, ny, nz]).astype(numpy.float32) * 1024

    laplacian = LaplacianKernel(backend='c')
    with Timer() as s_t:
        a = laplacian(input_grid)
    print("Specialized time {:0.3f}s".format(s_t.interval))
    print("a.shape {}".format(a.shape))

    if product(a.shape) < 200:
        print(a)

    # too slow to compare directly, this will apply python method to just a subset
    # py = LaplacianKernel(backend='python')
    # with Timer() as py_t:
    #     b = py(input_grid)
    # numpy.testing.assert_array_almost_equal(a, b, decimal=4)
    # print("PASSED")
    # print("Py time {:0.3f}".format(py_t.interval))
