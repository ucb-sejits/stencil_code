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

    def test_corner_points(self):
        stencil_grid = StencilGrid([3, 3])

        corners = [x for x in stencil_grid.corner_points()]
        self._are_lists_equal(corners, [(0, 0), (0, 2), (2, 0), (2, 2)])

        # print "corners"
        # for point in stencil_grid.corner_points():
        #     print(point)

    def test_edge_points(self):
        stencil_grid = StencilGrid([3, 3])

        edges = [x for x in stencil_grid.edge_points()]
        self._are_lists_equal(edges, [(0, 1), (2, 1), (1, 0), (1, 2)])
        # print "edges"
        # for point in stencil_grid.edge_points():
        #     print(point)

    def test_border_points(self):
        stencil_grid = StencilGrid([3, 3])

        border = [x for x in stencil_grid.border_points()]
        self._are_lists_equal(border, [(0, 1), (2, 1), (1, 0), (1, 2), (0, 0), (0, 2), (2, 0), (2, 2)])
        # print "border"
        # for point in stencil_grid.border_points():
        #     print(point)
