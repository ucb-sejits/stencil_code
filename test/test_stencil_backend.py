import unittest

from stencil_code.stencil_kernel2 import Stencil
from stencil_code.backend.omp import *
from ctree.transformations import PyBasicConversions


class TestStencilBackend(unittest.TestCase):
    def _check(self, actual, expected):
        self.assertEqual(str(actual), str(expected))

    @unittest.skip("not finished")
    def test_simple_transform(self):
        class OneDimStencil(Stencil):
            neighborhoods = [[-1, 0, 1],]

            def kernel(self, in_img, out_img):
                for x in self.interior_points(out_img):
                    for y in self.neighbors(x):
                        out_img[x] += in_img[y]

        kernel = OneDimStencil()
        kernel.should_unroll = False
        out_grid = StencilGrid([5])
        out_grid.ghost_depth = radius
        in_grid = StencilGrid([5])
        in_grid.ghost_depth = radius
        for x in range(0, 5):
            in_grid.data[x] = 1

        tree1 = ctree.get_ast(Kernel.kernel)
        tree2 = PyBasicConversions().visit(tree1)
        actual = StencilOmpTransformer([in_grid], out_grid, kernel).visit(tree2)


        self.assertEqual(actual, second)
