from stencil_code.neighborhood import Neighborhood

__author__ = 'chick'

import unittest
import numpy
import itertools

from stencil_code.halo_enumerator import HaloEnumerator
from stencil_code.stencil_kernel import Stencil, product


class TestHaloEnumerator(unittest.TestCase):
    def _are_lists_equal(self, list1, list2):
        self.assertEqual(sorted(list1), sorted(list2))

    def _are_lists_unequal(self, list1, list2):
        self.assertNotEqual(sorted(list1), sorted(list2))

    def test_constructor(self):
        with self.assertRaises(AssertionError) as context:
            HaloEnumerator([1, 1], [5, 5, 5])

        self.assertTrue("HaloEnumerator halo" in context.exception.args[0])

    def test_1_d(self):
        self._are_lists_equal(list(HaloEnumerator([1], [5])), [(0,), (4,)] )
        self._are_lists_equal(list(HaloEnumerator([2], [5])), [(0,), (1,), (3,), (4,)] )

    def test_n_d(self):
        for dimension in range(1, 7):
            for halo_size in range(1, 3):
                shape = [5 for _ in range(dimension)]
                halo = [halo_size for _ in range(dimension)]
                matrix = numpy.zeros(shape)
                elements = product(shape)
                dims = (range(0, dim) for dim in shape)
                all_indices = set([x for x in itertools.product(*dims)])

                neighborhood = [
                    Neighborhood.moore_neighborhood(radius=halo_size, dim=dimension, include_origin=False)
                ]

                class Kernel(Stencil):
                    neighborhoods = neighborhood

                    def kernel(self):
                        pass

                kernel = Kernel(backend='python', boundary_handling='zero')
                interior_set = set(list(kernel.interior_points(matrix)))
                halo_set = set(list(HaloEnumerator(halo, shape)))

                # print("halo_size {} dimension {} shape {} halo {}".format(
                #     halo_size, dimension, shape, halo
                # ))
                # print("all indices {}".format(all_indices))
                # print("halo_set {}".format(halo_set))
                # print("interior_set {}".format(interior_set))
                # print("intersection {}".format(interior_set.intersection(halo_set)))
                # print("outliers {}".format(halo_set.difference(all_indices)))

                self.assertTrue(len(interior_set.intersection(halo_set)) == 0)
                self.assertTrue(halo_set.issubset(all_indices))
                self.assertTrue(interior_set.issubset(all_indices))
                self.assertTrue(interior_set.union(halo_set) == all_indices)

    def test_with_kernel(self):
        for dimension in range(1, 7):
            for halo_size in range(1, 3):
                shape = [5 for _ in range(dimension)]
                halo = [halo_size for _ in range(dimension)]
                matrix = numpy.zeros(shape)
                elements = product(shape)
                dims = (range(0, dim) for dim in shape)
                all_indices = set([x for x in itertools.product(*dims)])

                neighborhood = [
                    Neighborhood.moore_neighborhood(radius=halo_size, dim=dimension, include_origin=False)
                ]

                class Kernel(Stencil):
                    neighborhoods = neighborhood

                    def kernel(self):
                        pass

                kernel = Kernel(backend='python', boundary_handling='zero')
                interior_set = set(list(kernel.interior_points(matrix)))
                halo_set = set(list(kernel.halo_points(matrix)))

                # print("halo_size {} dimension {} shape {} halo {}".format(
                #     halo_size, dimension, shape, halo
                # ))
                # print("all indices {}".format(all_indices))
                # print("halo_set {}".format(halo_set))
                # print("interior_set {}".format(interior_set))
                # print("intersection {}".format(interior_set.intersection(halo_set)))
                # print("outliers {}".format(halo_set.difference(all_indices)))

                self.assertTrue(len(interior_set.intersection(halo_set)) == 0)
                self.assertTrue(halo_set.issubset(all_indices))
                self.assertTrue(interior_set.issubset(all_indices))
                self.assertTrue(interior_set.union(halo_set) == all_indices)
