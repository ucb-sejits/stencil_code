import unittest

from stencil_code.backend.omp import *
from ctree.transformations import PyBasicConversions



class TestUnroll(unittest.TestCase):
    def _check(self, actual, expected):
        self.assertEqual(str(actual), str(expected))

    @unittest.skip("not finished")
    def test_simple_transform(self):
        class Kernel(StencilKernel):
            def kernel(self, in_img, out_img):
                for x in out_img.interior_points():
                    for y in in_img.neighbors(x, 1):
                        out_img[x] += in_img[y]

        kernel = Kernel()
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
