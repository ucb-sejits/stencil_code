__author__ = 'leonardtruong'
import unittest

import numpy
import numpy.testing
from kernels import TwoDHeatFlow, LaplacianKernel, SpecializedLaplacian27, BetterBilateralFilter

width = 128
height = 128


class TestCEndToEnd(unittest.TestCase):
    def setUp(self):
        self.in_grid = numpy.random.random([width, width]).astype(numpy.float32) * 1000

    def _check(self, test_kernel):
        outgrid1 = test_kernel(backend='ocl')(self.in_grid)
        outgrid2 = test_kernel(backend='python')(self.in_grid)
        try:
            numpy.testing.assert_array_almost_equal(outgrid1[1:-1, 1:-1], outgrid2[1:-1, 1:-1])
        except AssertionError:
            self.fail("Output grids not equal")

    def test_2d_heat(self):
        self._check(TwoDHeatFlow)

    def test_laplacian(self):
        self._check(LaplacianKernel)

    def test_laplacian27(self):
        in_grid = numpy.random.random([32, 32, 32]).astype(numpy.float32) * 255
        out_grid1 = SpecializedLaplacian27(backend='ocl')(in_grid)
        out_grid2 = SpecializedLaplacian27(backend='python')(in_grid)

        try:
            numpy.testing.assert_array_almost_equal(
                out_grid1[1:-1, 1:-1, 1:-1], out_grid2[1:-1, 1:-1, 1:-1])
        except AssertionError:
            self.fail("Output grids not equal")

    def test_bilateral_filter(self):
        in_grid = numpy.random.random([width, height]).astype(numpy.float32) * 255
        out_grid1 = BetterBilateralFilter(backend='ocl')(in_grid)
        out_grid2 = BetterBilateralFilter(backend='python')(in_grid)

        # print("o1 {}".format(out_grid1))
        print("o1 {}".format(out_grid1[3, 3:8]))
        print("o2 {}".format(out_grid2[3, 3:8]))
        try:
            numpy.testing.assert_array_almost_equal(
                out_grid1[3:-5, 3:-5], out_grid2[3:-5, 3:-5], decimal=1)
        except AssertionError:
            self.fail("Output grids not equal")
