__author__ = 'chick'

import unittest

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
