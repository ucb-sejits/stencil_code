import unittest

from stencil_code.stencil_grid import StencilGrid


class TestStencilGrid(unittest.TestCase):
    def _are_lists_equal(self, list1, list2):
        self.assertEqual(sorted(list1), sorted(list2))

    def _are_lists_unequal(self, list1, list2):
        self.assertNotEqual(sorted(list1), sorted(list2))

    def test_moore_neighborhood(self):
        stencil_grid = StencilGrid([2, 2])

        neighborhood = stencil_grid.moore_neighborhood()

        self._are_lists_equal(
            neighborhood,
            [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        )

        self._are_lists_unequal(
            neighborhood,
            [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        )

        neighborhood = stencil_grid.moore_neighborhood(include_origin=True)

        self._are_lists_equal(
            neighborhood,
            [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        )

        stencil_grid = StencilGrid([2, 2, 2])

        neighborhood = stencil_grid.moore_neighborhood()

        self._are_lists_equal(
            neighborhood,
            [
                (-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 0), (-1, 0, 1), (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
                (0, -1, -1), (0, -1, 0), (0, -1, 1), (0, 0, -1), (0, 0, 1), (0, 1, -1), (0, 1, 0), (0, 1, 1),
                (1, -1, -1), (1, -1, 0), (1, -1, 1), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, -1), (1, 1, 0), (1, 1, 1)
            ]
        )

        neighborhood = stencil_grid.moore_neighborhood(include_origin=True)

        self._are_lists_equal(
            neighborhood,
            [
                (-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 0), (-1, 0, 1), (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
                (0, -1, -1), (0, -1, 0), (0, -1, 1), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 1, -1), (0, 1, 0), (0, 1, 1),
                (1, -1, -1), (1, -1, 0), (1, -1, 1), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, -1), (1, 1, 0), (1, 1, 1)
            ]
        )

    def test_von_neuman_neighborhood(self):
        stencil_grid = StencilGrid([2, 2])

        neighborhood = stencil_grid.von_neuman_neighborhood()

        self._are_lists_equal(
            neighborhood,
            [(-1, 0), (0, -1), (0, 1), (1, 0)]
        )

        self._are_lists_unequal(
            neighborhood,
            [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
        )

        stencil_grid = StencilGrid([2, 2, 2])

        neighborhood = stencil_grid.von_neuman_neighborhood()

        self._are_lists_equal(
            neighborhood,
            [
                (-1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1), (0, 1, 0), (1, 0, 0),
            ]
        )
