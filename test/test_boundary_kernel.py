from stencil_code.boundary_kernel import BoundaryCopyKernel
from stencil_code.neighborhood import Neighborhood
from stencil_code.stencil_exception import StencilException

__author__ = 'chick'

import unittest
import numpy
import itertools
from operator import mul

from stencil_code.halo_enumerator import HaloEnumerator
from stencil_code.stencil_kernel import Stencil


class TestBoundaryKernel(unittest.TestCase):
    def _are_lists_equal(self, list1, list2):
        self.assertEqual(sorted(list1), sorted(list2))

    def _are_lists_unequal(self, list1, list2):
        self.assertNotEqual(sorted(list1), sorted(list2))

    def test_compute_local_group_size(self):
        shape = [102, 7]
        halo = [3, 3]
        grid = numpy.ones(shape).astype(numpy.float32)

        bk = BoundaryCopyKernel(halo, grid, 0)
        self.assertEqual(bk.global_size, [3, 7])
        self.assertEqual(bk.global_offset, [0, 0])

        bk = BoundaryCopyKernel(halo, grid, 1)
        self.assertEqual(bk.global_size, [96, 3])
        self.assertEqual(bk.global_offset, [3, 0])

    def test_exception_for_bad_halo_grid_values(self):
        with self.assertRaises(StencilException) as context:
            BoundaryCopyKernel([2, 3], numpy.ones([4, 423]), 1)
        self.assertTrue("can't span" in context.exception.message)

        with self.assertRaises(StencilException) as context:
            BoundaryCopyKernel([2, 3, 4], numpy.ones([4, 423]), 1)
        self.assertTrue("can't apply" in context.exception.message)

        with self.assertRaises(StencilException) as context:
            BoundaryCopyKernel([0, 3], numpy.ones([4, 423]), 1)
        self.assertTrue("can't be bigger" in context.exception.message)

    def test_virtual_global_size(self):
        grid = numpy.ones([17, 513])
        bk = BoundaryCopyKernel([2, 2], grid, 0)
        self.assertEqual(bk.global_size, [2, 513])
        self.assertEqual(bk.virtual_global_size, [2, 514])

        print("global_size {} local_size {} virtual_global_size {}".format(
            bk.global_size, bk.local_size, bk.virtual_global_size
        ))


