from stencil_code.stencil_kernel import StencilKernel
from stencil_code.stencil_grid import StencilGrid
import numpy as np
import unittest
import random

height = 50
stdev_d = 3
stdev_s = 70
radius = 1
width = 2*8 + radius * 2


class TestOclEndToEnd(unittest.TestCase):
    def setUp(self):
        out_grid1 = StencilGrid([width, width])
        out_grid1.ghost_depth = radius
        out_grid2 = StencilGrid([width, width])
        out_grid2.ghost_depth = radius
        in_grid = StencilGrid([width, width])
        in_grid.ghost_depth = radius

        for x in range(0, width):
            for y in range(0, width):
                in_grid.data[(x, y)] = random.random() * random.randint(0, 1000)

        for x in range(-radius, radius+1):
            for y in range(-radius, radius+1):
                in_grid.neighbor_definition[1].append((x, y))
        self.grids = (in_grid, out_grid1, out_grid2)

    def _check(self, test_kernel):
        in_grid, out_grid1, out_grid2 = self.grids
        test_kernel(backend='ocl', testing=True).kernel(in_grid, out_grid1)
        test_kernel(pure_python=True).kernel(in_grid, out_grid2)
        try:
            np.testing.assert_array_almost_equal(out_grid1.data,
                                                 out_grid2.data,
                                                 decimal=3)
        except AssertionError as e:
            self.fail("Output grids not equal: %s" % e.message)

    def test_simple_ocl_kernel(self):
        class Kernel(StencilKernel):
            def kernel(self, in_grid, out_grid):
                for x in out_grid.interior_points():
                    for y in in_grid.neighbors(x, 1):
                        out_grid[x] += in_grid[y]
        self._check(Kernel)

    def test_2d_heat(self):
        class Kernel(StencilKernel):
            def kernel(self, in_img, out_img):
                for x in out_img.interior_points():
                    # out_img[x] = in_img[x]
                    for y in in_img.neighbors(x, 0):
                        out_img[x] += 0.125 * in_img[y]
                    for z in in_img.neighbors(x, 1):
                        out_img[x] -= 0.125 * 2.0 * in_img[z]
        self._check(Kernel)
