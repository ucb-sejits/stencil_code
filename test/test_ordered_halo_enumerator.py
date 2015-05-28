from stencil_code.neighborhood import Neighborhood

__author__ = 'chick'

import unittest
import numpy
import itertools

from stencil_code.ordered_halo_enumerator import OrderedHaloEnumerator
from stencil_code.stencil_kernel import Stencil


class TestOrderedOrderedHaloEnumerator(unittest.TestCase):
    def _are_lists_equal(self, list1, list2):
        self.assertEqual(sorted(list1), sorted(list2))

    def _are_lists_unequal(self, list1, list2):
        self.assertNotEqual(sorted(list1), sorted(list2))

    def test_constructor(self):
        with self.assertRaises(AssertionError) as context:
            OrderedHaloEnumerator([1, 1], [5, 5, 5])

        self.assertTrue("OrderedHaloEnumerator halo" in context.exception.args[0])

    def test_1_d(self):
        self._are_lists_equal(list(OrderedHaloEnumerator([1], [5])), [(0,), (4,)])
        self._are_lists_equal(list(OrderedHaloEnumerator([2], [5])), [(0,), (1,), (3,), (4,)])

    @staticmethod
    def add_tuple(a, b):
        return tuple(x+y for x, y in zip(a, b))

    def test_n_d(self):
        for dimension in range(1, 7):
            for halo_size in range(1, 3):
                shape = [5 for _ in range(dimension)]
                halo = [halo_size for _ in range(dimension)]
                matrix = numpy.zeros(shape)
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
                halo_set = set(list(OrderedHaloEnumerator(halo, shape)))

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
                matrix = numpy.zeros(shape)
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

    def test_no_redundancy(self):
        surface_points = set()
        for point in OrderedHaloEnumerator([2, 2, 2, 2], [5, 5, 5, 5]):
            self.assertFalse(point in surface_points)
            surface_points.add(point)

    def test_point_to_surface_key(self):
        ohe = OrderedHaloEnumerator([1, 1, 1], [3, 3, 3])
        for halo_key in ohe.ordered_border_type_enumerator():
            for halo_point in ohe.surface_iterator(halo_key):
                self.assertEqual(halo_key, ohe.point_to_surface_key(halo_point))

    def test_neighbor_direction(self):
        ohe = OrderedHaloEnumerator([1, 1, 1], [3, 3, 3])
        for halo_key in ohe.ordered_border_type_enumerator():
            for halo_point in ohe.surface_iterator(halo_key):
                neighbor_direction = ohe.neighbor_direction(halo_key, halo_point)
                a_neighbor_point = TestOrderedOrderedHaloEnumerator.add_tuple(halo_point, neighbor_direction)
                neighbor_surface_key = ohe.point_to_surface_key(a_neighbor_point)

                self.assertLess(
                    ohe.order_of_surface_key(neighbor_surface_key), ohe.order_of_surface_key(halo_key),
                    "order of surface {} of a_neighbor_point {} = {} "
                    "is not less than order of the surface of {} of point {} = {}".format(
                        ohe.point_to_surface_key(a_neighbor_point), a_neighbor_point,
                        ohe.order_of_surface_key(a_neighbor_point),
                        halo_key, ohe.order_of_surface_key(halo_key), halo_point
                    )
                )
