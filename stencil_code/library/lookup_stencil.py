from __future__ import print_function

from stencil_code.stencil_kernel import Stencil
from stencil_code.neighborhood import Neighborhood
import numpy
import numpy.testing

import sys


class LookupStencil(Stencil):
    neighborhoods = [Neighborhood.moore_neighborhood(radius=1, dim=2, include_origin=False)]

    def kernel(self, in_grid, lookup_table, out_grid):
        for x in self.interior_points(out_grid):
            acc = 0
            for n in self.neighbors(x, 0):
                acc += in_grid[n]
            acc += (8 * in_grid[x])
            out_grid[x] = lookup_table[acc]


if __name__ == '__main__':  # pragma: no cover
    import logging
    logging.basicConfig(level=20)

    nx = 10 if len(sys.argv) <= 1 else int(sys.argv[1])
    ny = 10 if len(sys.argv) <= 2 else int(sys.argv[2])

    numpy.random.seed(0)
    # input_grid = numpy.random.randint(2, size=[10, 10])
    input_grid = numpy.zeros([nx, ny], dtype=numpy.int32)
    input_grid[5, :] = 1

    lut = numpy.array([
        0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 0, 0,
    ]).astype(numpy.int32)

    lookup_stencil = LookupStencil(backend='ocl', boundary_handling='zero')

    print(input_grid[:min(10, input_grid.shape[0]), :min(10, input_grid.shape[1])])
    for _ in range(5):
        input_grid = lookup_stencil(input_grid, lut)
        print(input_grid[:min(10, input_grid.shape[0]), :min(10, input_grid.shape[1])])
