from stencil_code.boundary_kernel import BoundaryCopyKernel
from stencil_code.neighborhood import Neighborhood

__author__ = 'chick'

import unittest
import numpy
import itertools
from operator import mul

from stencil_code.halo_enumerator import HaloEnumerator
from stencil_code.stencil_kernel import Stencil
from stencil_code.backend.ocl_tools import product, OclTools

class TestOclTools(unittest.TestCase):
    def _are_lists_equal(self, list1, list2):
        self.assertEqual(sorted(list1), sorted(list2))

    def _are_lists_unequal(self, list1, list2):
        self.assertNotEqual(sorted(list1), sorted(list2))

    def test_prod(self):
        self.assertTrue(product([1]) == 1)
        self.assertTrue(product([2, 3, 4]) == 24)

    def test_compute_local_group_size_1d(self):
        tools = OclTools()

        # chooses the minimum of shape / 2 and max local group size
        self.assertTrue(
            tools.compute_local_size_1d([100]) == 50,
            "when smaller than work group divide by 2"
        )
        print("ls1d {}".format(tools.compute_local_size_1d([1000])))

        self.assertTrue(
            tools.compute_local_size_1d([1000]) == 500,
            "when smaller than work group divide by 2"
        )

        self.assertTrue(
            tools.compute_local_size_1d([10000]) == 512,
            "when smaller than work group divide by 2"
        )

    def test_compute_local_group_size_2d(self):
        tools = OclTools()

        print("local_size {}".format(tools.compute_local_size_2d([1, 101])))
        print("local_size {}".format(tools.compute_local_size_2d([512, 101])))
        print("local_size {}".format(tools.compute_local_size_2d([512, 513])))
        print("local_size {}".format(tools.compute_local_size_2d([101, 1025])))
        print("local_size {}".format(tools.compute_local_size_2d([5120, 32])))

        shape = [102, 7]

        # print("local_size {}".format(tools.compute_local_size_2d(shape)))
