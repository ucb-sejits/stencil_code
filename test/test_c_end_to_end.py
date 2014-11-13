__author__ = 'leonardtruong'
import unittest

# from .examples.two_d_heat import TwoDHeatFlow
# from examples.laplacian import LaplacianKernel
# from examples.laplacian_27pt import SpecializedLaplacian27
# from examples.better_bilateral_filter import BetterBilateralFilter

from .kernels import TwoDHeatFlow, LaplacianKernel, SpecializedLaplacian27, BetterBilateralFilter

import numpy as np

stdev_d = 3
stdev_s = 70
radius = 1
width = 128
height = width


class TestCEndToEnd(unittest.TestCase):
    def setUp(self):
        self.in_grid = np.random.rand(width, width).astype(np.float32) * 1000

    def _check(self, test_kernel):
        outgrid1 = test_kernel(backend='c')(self.in_grid)
        outgrid2 = test_kernel(backend='python')(self.in_grid)
        try:
            np.testing.assert_array_almost_equal(outgrid1, outgrid2)
        except Exception:
            self.fail("Output grids not equal")

    def test_2d_heat(self):
        self._check(TwoDHeatFlow)

    def test_laplacian(self):
        self._check(LaplacianKernel)

    def test_laplacian(self):
        self._check(LaplacianKernel)

    def test_bilateral_filter(self):
        in_grid = np.random.rand(width, height).astype(np.float32) * 255
        out_grid1 = BetterBilateralFilter(backend='c')(in_grid)
        out_grid2 = BetterBilateralFilter(backend='python')(in_grid)

        try:
            np.testing.assert_array_almost_equal(out_grid1, out_grid2)
        except:
            self.fail("Output grids not equal")
