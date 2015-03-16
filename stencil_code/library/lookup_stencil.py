from __future__ import print_function

from stencil_code.stencil_kernel import Stencil, product
from stencil_code.neighborhood import Neighborhood
from ctree.util import Timer
import numpy
import numpy.testing

import sys


class LookupStencil(Stencil):
    neighborhoods = [Neighborhood.moore_neighborhood(radius=1, dim=2)]

    def kernel(self, in_grid, lut, out_grid):
        for x in self.interior_points(out_grid):
            acc = 0
            for n in self.neighbors(x, 0):
                acc += in_grid[n]
            out_grid[x] = lut[acc]


if __name__ == '__main__':  # pragma: no cover
    import logging
    logging.basicConfig(level=20)

    nx = 8 if len(sys.argv) <= 1 else int(sys.argv[1])
    ny = 32 if len(sys.argv) <= 2 else int(sys.argv[2])
    nz = 128 if len(sys.argv) <= 3 else int(sys.argv[3])

    numpy.random.seed(0)
    # input_grid = numpy.random.randint(2, size=[10, 10])
    input_grid = numpy.zeros([10, 10], dtype=numpy.int32)
    input_grid[5, :] = 1

    lut = numpy.array([[
        0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 0,
    ]]).astype(numpy.int32)

    lookup_stencil = LookupStencil(backend='c',boundary_handling='zero')

    print(input_grid)
    for _ in range(5):
        input_grid = lookup_stencil(input_grid, lut)
        print(input_grid)
