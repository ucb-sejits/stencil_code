# import unittest

# from stencil_code.stencil_grid import StencilGrid


# class TestStencilGrid(unittest.TestCase):
#     def _are_lists_equal(self, list1, list2):
#         self.assertEqual(sorted(list1), sorted(list2))

#     def _are_lists_unequal(self, list1, list2):
#         self.assertNotEqual(sorted(list1), sorted(list2))

#     def test_moore_neighborhood(self):
#         stencil_grid = StencilGrid([2, 2])

#         neighborhood = stencil_grid.moore_neighborhood()

#         self._are_lists_equal(
#             neighborhood,
#             [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
#         )

#         self._are_lists_unequal(
#             neighborhood,
#             [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
#         )

#         neighborhood = stencil_grid.moore_neighborhood(include_origin=True)

#         self._are_lists_equal(
#             neighborhood,
#             [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
#         )

#         stencil_grid = StencilGrid([2, 2, 2])

#         neighborhood = stencil_grid.moore_neighborhood()

#         self._are_lists_equal(
#             neighborhood,
#             [
#                 (-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 0), (-1, 0, 1), (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
#                 (0, -1, -1), (0, -1, 0), (0, -1, 1), (0, 0, -1), (0, 0, 1), (0, 1, -1), (0, 1, 0), (0, 1, 1),
#                 (1, -1, -1), (1, -1, 0), (1, -1, 1), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, -1), (1, 1, 0), (1, 1, 1)
#             ]
#         )

#         neighborhood = stencil_grid.moore_neighborhood(include_origin=True)

#         self._are_lists_equal(
#             neighborhood,
#             [
#                 (-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 0), (-1, 0, 1), (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
#                 (0, -1, -1), (0, -1, 0), (0, -1, 1), (0, 0, -1), (0, 0, 0), (0, 0, 1), (0, 1, -1), (0, 1, 0), (0, 1, 1),
#                 (1, -1, -1), (1, -1, 0), (1, -1, 1), (1, 0, -1), (1, 0, 0), (1, 0, 1), (1, 1, -1), (1, 1, 0), (1, 1, 1)
#             ]
#         )

#     def test_von_neuman_neighborhood(self):
#         stencil_grid = StencilGrid([2, 2])

#         neighborhood = stencil_grid.von_neuman_neighborhood()

#         self._are_lists_equal(
#             neighborhood,
#             [(-1, 0), (0, -1), (0, 1), (1, 0)]
#         )

#         self._are_lists_unequal(
#             neighborhood,
#             [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
#         )

#         stencil_grid = StencilGrid([2, 2, 2])

#         neighborhood = stencil_grid.von_neuman_neighborhood()

#         self._are_lists_equal(
#             neighborhood,
#             [
#                 (-1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1), (0, 1, 0), (1, 0, 0),
#             ]
#         )

#     def test_corner_points(self):
#         stencil_grid = StencilGrid([3, 3])

#         corners = [x for x in stencil_grid.corner_points()]
#         self._are_lists_equal(corners, [(0, 0), (0, 2), (2, 0), (2, 2)])

#         # print "corners"
#         # for point in stencil_grid.corner_points():
#         #     print(point)

#         import itertools
#         stencil_grid = StencilGrid([10, 9, 8, 7, 6])

#         corners = [x for x in stencil_grid.corner_points()]
#         ends = [[0, 9], [0, 8], [0, 7], [0, 6], [0, 5]]
#         self._are_lists_equal(corners, itertools.product(*ends))

#     def test_edge_points(self):
#         stencil_grid = StencilGrid([3, 3])

#         edges = [x for x in stencil_grid.edge_points()]
#         self._are_lists_equal(edges, [(0, 1), (2, 1), (1, 0), (1, 2)])
#         # print "edges"
#         # for point in stencil_grid.edge_points():
#         #     print(point)

#     def test_border_points(self):
#         stencil_grid = StencilGrid([3, 3])

#         border = [x for x in stencil_grid.border_points()]
#         # self._are_lists_equal(border, [(0, 1), (2, 1), (1, 0), (1, 2), (0, 0), (0, 2), (2, 0), (2, 2)])
#         # print "border"
#         # for point in stencil_grid.border_points():
#         #     print(point)

#         stencil_grid = StencilGrid([3, 3, 3])
#         border = sorted(list(stencil_grid.border_points()))
#         # print "corners %s" % sorted(list(stencil_grid.corner_points()))
#         # print "edges   %s" % sorted(list(stencil_grid.edge_points()))

#         border2 = sorted(list(stencil_grid.boundary_points()))
#         # print border2

#         # print("len border %d len border2 %d" % (len(border), len(border2)))
#         # print("len border %d len border2 %d" % (len(set(border)), len(set(border2))))
#         # print("sorted border  %s" % border)
#         # print("sorted border2 %s" % border2)
#         self._are_lists_equal(border, border2)

#         stencil_grid = StencilGrid([100, 99, 8, 7, 6])
#         border1 = sorted(list(stencil_grid.border_points()))
#         border2 = sorted(list(stencil_grid.boundary_points()))
#         # print("len border1 %d len border2 %d" % (len(border1), len(border2)))
#         # print("len border1 %d len border2 %d" % (len(set(border1)), len(set(border2))))
#         # print("sorted border1 %s" % border1)
#         # print("sorted border2 %s" % border2)
#         self._are_lists_equal(list(set(border1)), border2)



