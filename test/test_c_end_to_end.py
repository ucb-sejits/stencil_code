__author__ = 'leonardtruong'
import unittest

from kernels import TwoDHeatFlow, LaplacianKernel, SpecializedLaplacian27, BetterBilateralFilter

import numpy
import numpy.testing

width = 128
height = 64


class TestCEndToEnd(unittest.TestCase):
    def setUp(self):
        self.in_grid = numpy.random.random([width, width]).astype(numpy.float32) * 1000

    def _check(self, test_kernel):
        outgrid1 = test_kernel(backend='c')(self.in_grid)
        outgrid2 = test_kernel(backend='python')(self.in_grid)
        try:
            numpy.testing.assert_array_almost_equal(outgrid1, outgrid2)
        except Exception:
            self.fail("Output grids not equal")

    def test_2d_heat(self):
        self._check(TwoDHeatFlow)

    def test_laplacian(self):
        self._check(LaplacianKernel)

    def test_laplacian27(self):
        in_grid = numpy.random.random([32, 32, 32]).astype(numpy.float32) * 255
        out_grid1 = SpecializedLaplacian27(backend='c')(in_grid)
        out_grid2 = SpecializedLaplacian27(backend='python')(in_grid)

        try:
            numpy.testing.assert_array_almost_equal(out_grid1, out_grid2)
        except:
            self.fail("Output grids not equal")

    def test_bilateral_filter(self):
        in_grid = numpy.random.random([width, height]).astype(numpy.float32) * 255
        out_grid1 = BetterBilateralFilter(backend='ocl', sigma_d=1, sigma_i=70)(in_grid)
        out_grid2 = BetterBilateralFilter(backend='python')(in_grid)

        print(out_grid2)
        try:
            numpy.testing.assert_array_almost_equal(out_grid1, out_grid2)
        except:
            self.fail("Output grids not equal")
