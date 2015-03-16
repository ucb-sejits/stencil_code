import unittest

import numpy as np
from stencil_code.backend.c import StencilCTransformer
from stencil_code.neighborhood import Neighborhood

from stencil_code.stencil_kernel import Stencil
from stencil_code.backend.omp import *
from stencil_code.stencil_exception import StencilException
from ctree.transformations import PyBasicConversions


class LookupStencil(Stencil):
    neighborhoods = [Neighborhood.moore_neighborhood(radius=1, dim=2)]

    def kernel(self, in_grid, lut, out_grid):
        for x in self.interior_points(out_grid):
            acc = 0
            for n in self.neighbors(x, 0):
                acc += in_grid[n]
            out_grid[x] = lut[acc]


class TestStencilBackend(unittest.TestCase):

    def test_lookup_table(self):
        in_grid = np.zeros([10, 10])
        in_grid[5, :] = 1
        lookup_table = np.zeros([10])

        lookup_stencil = LookupStencil(backend='c')

        out_grid = lookup_stencil(in_grid, lookup_table)
        print(in_grid)
        print("X"*80)
        print(out_grid)



        print(lookup_stencil)