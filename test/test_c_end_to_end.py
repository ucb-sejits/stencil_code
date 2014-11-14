__author__ = 'leonardtruong'
import unittest

import numpy
import numpy.testing

from stencil_code.library.better_bilateral_filter import BetterBilateralFilter
from stencil_code.library.laplacian import LaplacianKernel
from stencil_code.library.laplacian_27pt import SpecializedLaplacian27
from stencil_code.library.two_d_heat import TwoDHeatFlow

width = 128
height = 64


class TestCEndToEnd(unittest.TestCase):
    def setUp(self):
        self.in_grid = numpy.random.random([width, width]).astype(numpy.float32) * 1000

    def _check(self, stencil_class_to_test, in_grid=None):
        if in_grid is None:
            in_grid = self.in_grid

        hp_stencil = stencil_class_to_test(backend='c')
        python_stencil = stencil_class_to_test(backend='python')
        hp_out_grid = hp_stencil(in_grid)
        python_out_grid = python_stencil(in_grid)

        interior_points_slice = tuple([slice(x, -x) for x in hp_stencil.ghost_depth])
        try:
            numpy.testing.assert_array_almost_equal(
                hp_out_grid[interior_points_slice],
                python_out_grid[interior_points_slice]
            )
        except AssertionError:
            self.fail("Output grids not equal")

    def test_2d_heat(self):
        in_grid = numpy.random.random([16, 16, 16]).astype(numpy.float32) * 1000
        self._check(TwoDHeatFlow, in_grid)

    def test_laplacian(self):
        in_grid = numpy.random.random([32, 32, 32]).astype(numpy.float32) * 1000
        self._check(LaplacianKernel, in_grid)

    def test_laplacian27(self):
        in_grid = numpy.random.random([32, 32, 32]).astype(numpy.float32) * 255

        out_grid1 = SpecializedLaplacian27(backend='c')(in_grid)
        out_grid2 = SpecializedLaplacian27(backend='python')(in_grid)

        try:
            numpy.testing.assert_array_almost_equal(out_grid1, out_grid2)
        except AssertionError:
            self.fail("Output grids not equal")

    def test_bilateral_filter(self):
        in_grid = numpy.random.random([width, height]).astype(numpy.float32) * 255
        out_grid1 = BetterBilateralFilter(backend='c')(in_grid)
        out_grid2 = BetterBilateralFilter(backend='python')(in_grid)

        try:
            numpy.testing.assert_array_almost_equal(out_grid1[5:-5, 5:-5], out_grid2[5:-5, 5:-5], decimal=3)
        except AssertionError:
            self.fail("Output grids not equal")
