from stencil_code.boundary_kernel import BoundaryCopyKernel
from stencil_code.neighborhood import Neighborhood

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

        print("grid {} halo {} global {} local {}".format(
            shape, halo, bk.global_size, bk.local_size
        ))

    def test_exception_for_bad_halo_grid_values(self):
        BoundaryCopyKernel([2, 3], numpy.ones([4, 423]), 1)


