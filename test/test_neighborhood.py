__author__ = 'chick'

import unittest
import numpy

from stencil_code.neighborhood import Neighborhood


class TestNeighborhood(unittest.TestCase):
    def _are_lists_equal(self, list1, list2):
        self.assertEqual(sorted(list1), sorted(list2))

    def _are_lists_unequal(self, list1, list2):
        self.assertNotEqual(sorted(list1), sorted(list2))

    def test_origin(self):
        self.assertEqual(tuple([0]), Neighborhood.origin_of_dim(1))
        self.assertEqual((0, 0), Neighborhood.origin_of_dim(2))
        self.assertEqual((0, 0, 0), Neighborhood.origin_of_dim(3))
        self.assertEqual((0, 0, 0, 0), Neighborhood.origin_of_dim(4))
        self.assertEqual((0, 0, 0, 0, 0), Neighborhood.origin_of_dim(5))

    def test_von_neuman_neighborhood(self):
        default = Neighborhood.von_neuman_neighborhood()

        self._are_lists_equal(
            default,
            [(-1, 0), (0, -1), (0, 1), (1, 0), (0, 0)]
        )

        default_without_origin = Neighborhood.von_neuman_neighborhood(include_origin=False)
        self._are_lists_equal(
            default_without_origin,
            [(-1, 0), (0, -1), (0, 1), (1, 0)]
        )

        neighborhood_3_2 = Neighborhood.von_neuman_neighborhood(3, 2)
        self._are_lists_equal(
            neighborhood_3_2,
            [
                (-3, 0), (-2, -1), (-2, 0), (-2, 1),
                (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
                (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3),
                (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
                (2, -1), (2, 0), (2, 1), (3, 0)
            ]
        )

        neighborhood_1_3 = Neighborhood.von_neuman_neighborhood(1, 3)
        self._are_lists_equal(
            neighborhood_1_3,
            [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)]
        )

    def test_moore_neighborhood(self):
        default = Neighborhood.moore_neighborhood()

        self._are_lists_equal(
            default,
            [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        )

        default_without_origin = Neighborhood.moore_neighborhood(include_origin=False)
        self._are_lists_equal(
            default_without_origin,
            [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        )

        neighborhood_3_2 = Neighborhood.moore_neighborhood(3, 2)
        self._are_lists_equal(
            neighborhood_3_2,
            [
                (-3, -3), (-3, -2), (-3, -1), (-3, 0), (-3, 1), (-3, 2), (-3, 3),
                (-2, -3), (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-2, 3),
                (-1, -3), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3),
                (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3),
                (1, -3), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (1, 3),
                (2, -3), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (2, 3),
                (3, -3), (3, -2), (3, -1), (3, 0), (3, 1), (3, 2), (3, 3)
            ]
        )

        neighborhood_1_3 = Neighborhood.moore_neighborhood(1, 3)
        self._are_lists_equal(
            neighborhood_1_3,
            [
                (-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 0),
                (-1, 0, 1), (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
                (0, -1, -1), (0, -1, 0), (0, -1, 1), (0, 0, -1), (0, 0, 0), (0, 0, 1),
                (0, 1, -1), (0, 1, 0), (0, 1, 1),
                (1, -1, -1), (1, -1, 0), (1, -1, 1), (1, 0, -1), (1, 0, 0),
                (1, 0, 1), (1, 1, -1), (1, 1, 0), (1, 1, 1)
            ]
        )

    def test_compute_from_indices(self):
        coff = [
            [1, 1, 4],
            [1, 4, 8],
            [2, 8, 16],
        ]

        n, c, h = Neighborhood.compute_from_indices(coff)
        self.assertEqual(len(n), 9)
        self.assertEqual(len(c), 9)
        self.assertEqual(h[0][0], 1, "halo in negative direction of dim 1 is 1")
        self.assertEqual(h[0][1], 1, "halo in negative direction of dim 1 is 1")
        self.assertEqual(h[1][0], 1, "halo in positive direction of dim 2 is 1")
        self.assertEqual(h[1][1], 1, "halo in positive direction of dim 2 is 1")

        self._are_lists_equal(n, Neighborhood.moore_neighborhood(1, 2))

        coff = numpy.array(coff)  # do this just so we can index with a tuple
        for index, value in enumerate(n):
            value = tuple([value[0] + 1, value[1] + 1])  # add one because indices have become offsets
            self.assertEqual(c[index], coff[value])
        self.assertItemsEqual(c, [1, 1, 4, 1, 4, 8, 2, 8, 16])


        coff = [
            [0, 1, 0],
            [1, 4, 8],
            [0, 8, 0],
        ]

        n, c, h = Neighborhood.compute_from_indices(coff)
        self.assertEqual(len(n), 5, "length is 5 because indices with coefficient zero are dropped")
        self.assertEqual(len(c), 5)
        self.assertEqual(h[0][0], 1, "halo in negative direction of dim 1 is 1")
        self.assertEqual(h[0][1], 1, "halo in negative direction of dim 1 is 1")
        self.assertEqual(h[1][0], 1, "halo in positive direction of dim 2 is 1")
        self.assertEqual(h[1][1], 1, "halo in positive direction of dim 2 is 1")

        coff = numpy.array(coff)  # do this just so we can index with a tuple
        for index, value in enumerate(n):
            value = tuple([value[0] + 1, value[1] + 1])  # add one because indices have become offsets
            self.assertEqual(c[index], coff[value])
        self._are_lists_equal(n, Neighborhood.von_neuman_neighborhood(1, 2))
