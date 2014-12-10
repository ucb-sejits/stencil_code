import unittest

from stencil_code.stencil_exception import StencilException
from stencil_code.stencil_kernel import Stencil


class TestStencilKernel(unittest.TestCase):
    def test_no_kernel_exception(self):
        class BadKernel(Stencil):
            pass
        self.assertRaises(StencilException, BadKernel)

    def test_bad_boundary_handling(self):
        class Kernel(Stencil):
            neighborhoods = [[(0, 0)]]
            def kernel(self, *args):
                pass

        with self.assertRaises(StencilException) as context:
            _ = Kernel(boundary_handling='dog')

        self.assertTrue("boundary handling value" in context.exception.args[0])

    def test_no_neighborhood(self):
        class NoNeighborhood(Stencil):
            def kernel(self):
                pass

        with self.assertRaises(StencilException) as context:
            NoNeighborhood()

        self.assertTrue("neighborhoods must be defined" in context.exception.args[0])

    def test_bad_neighborhood_id(self):
        class OneNeighborhood(Stencil):
            neighborhoods = [[(0,0)]]

            def kernel(self):
                pass

        one_neighborhood = OneNeighborhood()
        point = (0, 1)
        neighborhood_id = 2

        with self.assertRaises(StencilException) as context:
            next(one_neighborhood.neighbors(point, neighborhood_id))

        self.assertTrue("Undefined neighborhood identifier" in context.exception.args[0])




