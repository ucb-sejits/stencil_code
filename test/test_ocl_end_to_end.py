import numpy as np
import unittest
from kernels import SimpleKernel, TwoDHeatKernel, LaplacianKernel, \
    BilatKernel, gaussian1, gaussian2

stdev_d = 3
stdev_s = 70
radius = 1
width = 64 + radius * 2
height = width

class TestOclEndToEnd(unittest.TestCase):
    def setUp(self):
        self.in_grid = np.random.rand(width, width).astype(np.float32) * 1000

    def _check(self, test_kernel):
        out_grid1 = test_kernel(backend='ocl',
                                testing=True)(self.in_grid)
        out_grid2 = test_kernel(backend='python')(self.in_grid)
        try:
            np.testing.assert_array_almost_equal(out_grid1, out_grid2,
                                                 decimal=3)
        except AssertionError as e:
            self.fail("Output grids not equal: %s" % e.message)

    def test_simple_ocl_kernel(self):
        self._check(SimpleKernel)

    def test_2d_heat(self):
        self._check(TwoDHeatKernel)

    def test_laplacian(self):
        self._check(LaplacianKernel)

    def test_bilateral_filter(self):
        in_grid = np.random.rand(width, height).astype(np.float32) * 255

        out_grid1 = BilatKernel(backend='ocl', testing=True)(
            in_grid, gaussian1, gaussian2)
        out_grid2 = BilatKernel(backend='python')(
            in_grid, gaussian1, gaussian2)
        try:
            np.testing.assert_array_almost_equal(out_grid1, out_grid2)
        except AssertionError as e:
            self.fail("Output grids not equal: %s" % e.message)
