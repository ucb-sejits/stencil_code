__author__ = 'leonardtruong'

from kernels import SimpleKernel, TwoDHeatKernel, LaplacianKernel, \
    BilatKernel, gaussian1, gaussian2
import numpy as np
import unittest

stdev_d = 3
stdev_s = 70
radius = 1
width = 64 + radius * 2
height = width


class TestCEndToEnd(unittest.TestCase):
    def setUp(self):
        self.in_grid = np.random.rand(width, width).astype(np.float32) * 1000

    def _check(self, test_kernel):
        outgrid1 = test_kernel(backend='c', testing=True)(self.in_grid)
        outgrid2 = test_kernel(backend='python')(self.in_grid)
        try:
            np.testing.assert_array_almost_equal(outgrid1, outgrid2)
        except:
            self.fail("Output grids not equal")

    def test_simple_kernel(self):
        self._check(SimpleKernel)

    def test_2d_heat(self):
        self._check(TwoDHeatKernel)

    def test_laplacian(self):
        self._check(LaplacianKernel)

    def test_bilateral_filter(self):
        in_grid = np.random.rand(width, height).astype(np.float32) * 255
        out_grid1 = BilatKernel(backend='c', testing=True)(
            in_grid, gaussian1, gaussian2)
        out_grid2 = BilatKernel(backend='python')(
            in_grid, gaussian1, gaussian2)
        try:
            np.testing.assert_array_almost_equal(out_grid1, out_grid2)
        except:
            self.fail("Output grids not equal")
