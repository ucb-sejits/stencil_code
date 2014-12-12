__author__ = 'chick'

import unittest

import numpy
from ctree.c.nodes import If, Constant

from stencil_code.backend.ocl_boundary_copier import OclBoundaryCopier
from stencil_code.stencil_exception import StencilException


class TestBoundaryKernel(unittest.TestCase):
    def _are_lists_equal(self, list1, list2):
        self.assertEqual(sorted(list1), sorted(list2))

    def _are_lists_unequal(self, list1, list2):
        self.assertNotEqual(sorted(list1), sorted(list2))

    def __exception_contains(self, str, exception):
        return str in exception.__str__()

    def test_compute_local_group_size(self):
        shape = [102, 7]
        halo = [3, 3]
        grid = numpy.ones(shape).astype(numpy.float32)

        bk = OclBoundaryCopier(halo, grid, 0)
        self.assertEqual(bk.global_size, [3, 7])
        self.assertEqual(bk.global_offset, [0, 0])

        bk = OclBoundaryCopier(halo, grid, 1)
        self.assertEqual(bk.global_size, [96, 3])
        self.assertEqual(bk.global_offset, [3, 0])

    def test_exception_for_bad_halo_grid_values(self):
        with self.assertRaises(StencilException) as context:
            OclBoundaryCopier([2, 3], numpy.ones([4, 423]), 1)
        self.assertTrue(self.__exception_contains("can't span", context.exception))

        with self.assertRaises(StencilException) as context:
            OclBoundaryCopier([2, 3, 4], numpy.ones([4, 423]), 1)
        self.assertTrue(self.__exception_contains("can't apply", context.exception))

        with self.assertRaises(StencilException) as context:
            OclBoundaryCopier([0, 3], numpy.ones([4, 423]), 1)
        self.assertTrue(self.__exception_contains("can't be bigger", context.exception))

    def test_virtual_global_size(self):
        grid = numpy.ones([17, 513])
        bk = OclBoundaryCopier([2, 2], grid, 0)
        print("global_size {} local_size {} virtual_global_size {}".format(
            bk.global_size, bk.local_size, bk.virtual_global_size
        ))
        self.assertEqual(bk.global_size, [2, 513])
        self.assertEqual(bk.virtual_global_size, [2, 514])

    def test_gen_index_in_bounds_conditional(self):
        bk = OclBoundaryCopier([2, 2, 2], numpy.ones([512, 512, 513]), 1)
        bk.global_size = [3, 3, 3]
        bk.local_size = [4, 4, 4]
        self.assertTrue(
            type(bk.gen_index_in_bounds_conditional(Constant(1))) == If
        )
        with self.assertRaises(StencilException) as context:
            _ = OclBoundaryCopier([2, 2], numpy.ones([512, 512]), dimension=7)
        self.assertTrue("too big" in context.exception.args[0])

        bk = OclBoundaryCopier([2, 2], numpy.ones([512, 512]), 0)
        self.assertTrue(
            type(bk.gen_index_in_bounds_conditional(Constant(1))) == Constant
        )
        bk = OclBoundaryCopier([2, 2], numpy.ones([512, 513]), 0)
        self.assertTrue(
            type(bk.gen_index_in_bounds_conditional(Constant(1))) == If
        )
        bk = OclBoundaryCopier([2, 2], numpy.ones([512, 513]), 1)
        self.assertTrue(
            type(bk.gen_index_in_bounds_conditional(Constant(1))) == Constant
        )

        bk = OclBoundaryCopier([2, 2], numpy.ones([512, 513]), 1)
        bk.global_size = [3, 3]
        bk.local_size = [4, 4]
        self.assertTrue(
            type(bk.gen_index_in_bounds_conditional(Constant(1))) == If
        )
