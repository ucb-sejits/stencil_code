__author__ = 'chickmarkley'
import unittest

import numpy
import numpy.testing

from stencil_code.library.better_bilateral_filter import BetterBilateralFilter
from stencil_code.library.laplacian import LaplacianKernel
from stencil_code.library.laplacian_27pt import SpecializedLaplacian27
from stencil_code.library.two_d_heat import TwoDHeatFlow


class TestOclEndToEnd(unittest.TestCase):
    backend_to_test = 'ocl'
    backend_to_compare = 'c'

    def _compare_grids(self, stencil, grid1, grid2):
        interior_points_slice = stencil.interior_points_slice()
        # print("interior_points_slice {} of {}".format(interior_points_slice, stencil))
        numpy.testing.assert_array_almost_equal(
            grid1[interior_points_slice],
            grid2[interior_points_slice]
        )
        return
        try:
            numpy.testing.assert_array_almost_equal(
                grid1[interior_points_slice],
                grid2[interior_points_slice],
                decimals=3
            )
        except AssertionError:
            self.fail("Output grids not equal slice is {} for {}".format(interior_points_slice, stencil))

    def _check(self, stencil_class_to_test, in_grid, coefficients=None):
        hp_stencil = stencil_class_to_test(backend=TestOclEndToEnd.backend_to_test)
        compare_stencil = stencil_class_to_test(backend=TestOclEndToEnd.backend_to_compare)

        if coefficients is None:
            hp_out_grid = hp_stencil(in_grid)
            compare_grid = compare_stencil(in_grid)
        else:
            hp_out_grid = hp_stencil(in_grid, coefficients)
            compare_grid = compare_stencil(in_grid, coefficients)

        self._compare_grids(hp_stencil, hp_out_grid, compare_grid)

    def test_2d_heat(self):
        in_grid = numpy.random.random([16, 16, 16]).astype(numpy.float32) * 1000
        self._check(TwoDHeatFlow, in_grid)

    def test_laplacian(self):
        in_grid = numpy.random.random([32, 32, 32]).astype(numpy.float32) * 1000
        self._check(LaplacianKernel, in_grid)

    def test_laplacian27(self):
        in_grid = numpy.random.random([32, 32, 32]).astype(numpy.float32) * 255
        coefficients = numpy.array([1.0, 0.5, 0.25, 0.125]).astype(numpy.float32)
        self._check(SpecializedLaplacian27, in_grid, coefficients)

    def test_bilateral_filter(self):
        in_grid = numpy.random.random([64, 32]).astype(numpy.float32) * 255
        self._check(BetterBilateralFilter, in_grid)
