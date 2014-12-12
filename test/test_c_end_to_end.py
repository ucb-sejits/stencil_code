__author__ = 'leonardtruong'
import unittest

import numpy
import numpy.testing

from stencil_code.library.basic_convolution import ConvolutionFilter
from stencil_code.library.bilateral_filter import BilateralFilter, gaussian
from stencil_code.library.better_bilateral_filter import BetterBilateralFilter
from stencil_code.library.diagnostic_stencil import DiagnosticStencil
from stencil_code.library.jacobi_stencil import Jacobi
from stencil_code.library.laplacian import LaplacianKernel
from stencil_code.library.laplacian_27pt import SpecializedLaplacian27
from stencil_code.library.two_d_heat import TwoDHeatFlow

import logging
logging.basicConfig(level=20)


class TestCEndToEnd(unittest.TestCase):
    backend_to_test = 'c'
    backend_to_compare = 'python'

    def _compare_grids(self, stencil, grid1, grid2):
        interior_points_slice = stencil.interior_points_slice()
        try:
            numpy.testing.assert_array_almost_equal(
                grid1[interior_points_slice],
                grid2[interior_points_slice]
            )
        except AssertionError:
            self.fail("Output grids not equal")

    def _check(self, stencil_class_to_test, *args):
        hp_stencil = stencil_class_to_test(backend=TestCEndToEnd.backend_to_test)
        compare_stencil = stencil_class_to_test(backend=TestCEndToEnd.backend_to_compare)

        hp_out_grid = hp_stencil(*args)
        compare_grid = compare_stencil(*args)

        self._compare_grids(hp_stencil, hp_out_grid, compare_grid)

    def test_2d_heat(self):
        in_grid = numpy.random.random([16, 16, 16]).astype(numpy.float32) * 1000
        self._check(TwoDHeatFlow, in_grid)

    def test_convolution(self):
        in_grid = numpy.ones([32, 32]).astype(numpy.float32)
        stencil = numpy.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, -4, 1, 3],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 7, 1],
            ]
        )
        hp_stencil = ConvolutionFilter(convolution_array=stencil,
                                       backend=TestCEndToEnd.backend_to_test)
        compare_stencil = ConvolutionFilter(convolution_array=stencil,
                                            backend=TestCEndToEnd.backend_to_compare)

        hp_out_grid = hp_stencil(in_grid)
        compare_grid = compare_stencil(in_grid)

        self._compare_grids(hp_stencil, hp_out_grid, compare_grid)

    def test_diagnostic_kernel(self):
        in_grid = numpy.random.random([16, 16]).astype(numpy.float32) * 1000
        self._check(DiagnosticStencil, in_grid)

    def test_jacobi(self):
        in_grid = numpy.random.random([128, 128]).astype(numpy.float32) * 1000
        self._check(Jacobi, in_grid)

    def test_laplacian(self):
        in_grid = numpy.random.random([32, 32, 32]).astype(numpy.float32) * 1000
        self._check(LaplacianKernel, in_grid)

    def test_laplacian27(self):
        in_grid = numpy.random.random([32, 32, 32]).astype(numpy.float32) * 255
        coefficients = numpy.array([1.0, 0.5, 0.25, 0.125]).astype(numpy.float32)
        self._check(SpecializedLaplacian27, in_grid, coefficients)

    def test_bilateral_filter(self):
        gaussian1 = gaussian(3, 18)
        gaussian2 = gaussian(70, 256)

        in_grid = numpy.random.random([64, 32]).astype(numpy.float32) * 255
        self._check(BilateralFilter, in_grid, gaussian1, gaussian2)

    def test_better_bilateral_filter(self):
        in_grid = numpy.random.random([64, 32]).astype(numpy.float32) * 255
        self._check(BetterBilateralFilter, in_grid)
