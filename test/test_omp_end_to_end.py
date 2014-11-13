import unittest

import numpy
import numpy.testing
from nose.plugins.attrib import attr
from kernels import TwoDHeatFlow, LaplacianKernel, BetterBilateralFilter

stdev_d = 3
stdev_s = 70
radius = 1
width = 64 + radius * 2
height = width


class TestOmpEndToEnd(unittest.TestCase):
    def setUp(self):
        self.in_grid = numpy.random.random(width, width).astype(numpy.float32) * 1000

    def _check(self, test_kernel):
        out_grid1 = test_kernel(backend='omp',
                                testing=True)(self.in_grid)
        out_grid2 = test_kernel(backend='python')(self.in_grid)
        try:
            numpy.testing.assert_array_almost_equal(out_grid1, out_grid2)
        except AssertionError:
            self.fail("Output grids not equal")

    @attr('omp')
    def test_2d_heat(self):
        self._check(TwoDHeatFlow)

    @attr('omp')
    def test_laplacian(self):
        self._check(LaplacianKernel)

    @attr('omp')
    def test_bilateral_filter(self):
        in_grid = numpy.random.random(width, height).astype(numpy.float32) * 255
        out_grid1 = BetterBilateralFilter(backend='omp')(in_grid)
        out_grid2 = BetterBilateralFilter(backend='python')(in_grid)

        try:
            numpy.testing.assert_array_almost_equal(out_grid1, out_grid2)
        except AssertionError:
            self.fail("Output grids not equal")
