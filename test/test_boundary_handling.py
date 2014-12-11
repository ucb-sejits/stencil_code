import unittest
from nose.tools import assert_list_equal
import numpy
import numpy.testing
from stencil_code.library.diagnostic_stencil import DiagnosticStencil
from stencil_code.library.laplacian import LaplacianKernel
from stencil_code.neighborhood import Neighborhood

from stencil_code.stencil_kernel import Stencil


class TestBoundaryHandling(unittest.TestCase):
    def test_python_clamping(self):
        class Clamper(Stencil):
            neighborhoods = [Neighborhood.von_neuman_neighborhood(radius=1, dim=2)]

            def kernel(self, in_grid, out_grid):
                for p in self.interior_points(out_grid):
                    for n in self.neighbors(p, 0):
                        out_grid[p] += in_grid[n]

        clamper = Clamper(backend='python', boundary_handling='clamp')
        in_grid = numpy.ones([10, 10])

        clamper.current_shape = in_grid.shape   # this is ordinarily set by interior_points_loop

        assert_list_equal(
            [x for x in clamper.neighbors((9, 9), 0)],
            [(8, 9), (9, 8), (9, 9), (9, 9), (9, 9)],
            "clamping around (9, 9) should keep all values between 0 and 9 inclusive"
        )
        assert_list_equal(
            [x for x in clamper.neighbors((0, 0), 0)],
            [(0, 0), (0, 0), (0, 0), (0, 1), (1, 0)],
            "clamping around (0, 0) should keep all between 0 and 9 inclusive"
        )
        assert_list_equal(
            [x for x in clamper.neighbors((0, 0), 0)],
            [(0, 0), (0, 0), (0, 0), (0, 1), (1, 0)],
            "clamping around (0, 9) should keep all between 0 and 9 inclusive"
        )
        assert_list_equal(
            [x for x in clamper.neighbors((3, 3), 0)],
            [(2, 3), (3, 2), (3, 3), (3, 4), (4, 3)],
            "neighborhoods around interior points are not clamped"
        )

    def test_clamped(self):
        """
        zero boundary handling should just leave zero's in grid halo
        :return:
        """
        in_grid = numpy.ones([10, 10])

        python_clamped_kernel = DiagnosticStencil(backend='python', boundary_handling='clamp')
        c_clamped_kernel = DiagnosticStencil(backend='c', boundary_handling='clamp')
        python_clamped_out = python_clamped_kernel(in_grid)

        # import logging
        # logging.basicConfig(level=20)
        c_clamped_out = c_clamped_kernel(in_grid)

        numpy.testing.assert_array_almost_equal(python_clamped_out, c_clamped_out, decimal=4)
        self.assertTrue(python_clamped_out[0, 0] == 30)

        python_unclamped_kernel = DiagnosticStencil(backend='python', boundary_handling='zero')
        c_unclamped_kernel = DiagnosticStencil(backend='c', boundary_handling='zero')
        python_unclamped_out = python_unclamped_kernel(in_grid)
        c_unclamped_out = c_unclamped_kernel(in_grid)

        # print(python_unclamped_out)
        numpy.testing.assert_array_almost_equal(python_unclamped_out, c_unclamped_out, decimal=4)
        self.assertTrue(python_unclamped_out[0, 0] == 0)

    def test_copied_for_python(self):
        in_grid = numpy.ones([5, 5]).astype(numpy.float32)
        python_copy_boundary_kernel = DiagnosticStencil(backend='python', boundary_handling='copy')
        copy_out_grid = python_copy_boundary_kernel(in_grid)

        compare_list = [1. for _ in range(5)]
        assert_list_equal(list(copy_out_grid[0]), compare_list)
        assert_list_equal(list(copy_out_grid[4]), compare_list)
        assert_list_equal(list(copy_out_grid[:][0]), compare_list)
        assert_list_equal(list(copy_out_grid[:][4]), compare_list)

        python_clamp_boundary_kernel = DiagnosticStencil(backend='python', boundary_handling='clamp')
        copy_out_grid = python_clamp_boundary_kernel(in_grid)

        compare_list = [30. for _ in range(5)]
        assert_list_equal(list(copy_out_grid[0]), compare_list)
        assert_list_equal(list(copy_out_grid[4]), compare_list)
        assert_list_equal(list(copy_out_grid[:][0]), compare_list)
        assert_list_equal(list(copy_out_grid[:][4]), compare_list)

    def test_copied_for_c(self):
        # import logging
        # logging.basicConfig(level=20)
        in_grid = numpy.ones([5, 5]).astype(numpy.float32)
        python_copy_boundary_kernel = DiagnosticStencil(backend='c', boundary_handling='copy')
        copy_out_grid = python_copy_boundary_kernel(in_grid)

        compare_list = [1. for _ in range(5)]
        assert_list_equal(list(copy_out_grid[0]), compare_list)
        assert_list_equal(list(copy_out_grid[4]), compare_list)
        assert_list_equal(list(copy_out_grid[:][0]), compare_list)
        assert_list_equal(list(copy_out_grid[:][4]), compare_list)

        python_clamp_boundary_kernel = DiagnosticStencil(backend='c', boundary_handling='clamp')
        copy_out_grid = python_clamp_boundary_kernel(in_grid)

        compare_list = [30. for _ in range(5)]
        assert_list_equal(list(copy_out_grid[0]), compare_list)
        assert_list_equal(list(copy_out_grid[4]), compare_list)
        assert_list_equal(list(copy_out_grid[:][0]), compare_list)
        assert_list_equal(list(copy_out_grid[:][4]), compare_list)

    def test_copied_for_ocl(self):
        # import logging
        # logging.basicConfig(level=20)
        size = 8
        in_grid = numpy.ones([size, size]).astype(numpy.float32)
        copy_boundary_kernel = DiagnosticStencil(backend='ocl', boundary_handling='copy')
        copy_out_grid = copy_boundary_kernel(in_grid)

        compare_list = [1. for _ in range(size)]
        assert_list_equal(list(copy_out_grid[0]), compare_list)
        assert_list_equal(list(copy_out_grid[-1]), compare_list)
        assert_list_equal(list(copy_out_grid[:][0]), compare_list)
        assert_list_equal(list(copy_out_grid[:][-1]), compare_list)

        clamp_boundary_kernel = DiagnosticStencil(backend='ocl', boundary_handling='clamp')
        copy_out_grid = clamp_boundary_kernel(in_grid)

        compare_list = [30. for _ in range(size)]
        assert_list_equal(list(copy_out_grid[0]), compare_list)
        assert_list_equal(list(copy_out_grid[-1]), compare_list)
        assert_list_equal(list(copy_out_grid[:][0]), compare_list)
        assert_list_equal(list(copy_out_grid[:][-1]), compare_list)

    def test_copied_for_ocl_1d(self):
        # import logging
        # logging.basicConfig(level=20)
        class Stencil1d(Stencil):
            neighborhoods = [[(-1,), (0,), (1,)]]

            def kernel(self, in_grid, out_grid):
                for x in self.interior_points(out_grid):
                    for y in self.neighbors(x, 0):
                        out_grid[x] += 2 * in_grid[y]

        size = 8
        input_grid = numpy.ones(size).astype(numpy.float32)
        copy_boundary_kernel = Stencil1d(backend='ocl', boundary_handling='copy')
        copy_out_grid = copy_boundary_kernel(input_grid)

        self.assertEqual(copy_out_grid[0], 1.0)
        self.assertEqual(copy_out_grid[1], 6.0)
        self.assertEqual(copy_out_grid[-2], 6.0)
        self.assertEqual(copy_out_grid[-1], 1.0)

    def test_copied_for_ocl_3d(self):
        # import logging
        # logging.basicConfig(level=20)

        size = [8, 8, 8]
        input_grid = numpy.ones(size).astype(numpy.float32)
        copy_boundary_kernel = LaplacianKernel(backend='ocl', boundary_handling='copy')
        copy_out_grid = copy_boundary_kernel(input_grid)

        for point in copy_boundary_kernel.interior_points(copy_out_grid):
            self.assertEqual(copy_out_grid[point], 2.0)

        for point in copy_boundary_kernel.halo_points(copy_out_grid):
            self.assertEqual(copy_out_grid[point], 1.0)
