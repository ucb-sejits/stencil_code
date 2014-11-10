import unittest

from stencil_code.StencilException import StencilException
from stencil_code.stencil_kernel2 import Stencil


class TestStencilKernel(unittest.TestCase):
    def test_no_kernel_exception(self):
        class BadKernel(Stencil):
            pass
        self.assertRaises(StencilException, BadKernel)

    def test_no_neighborhood(self):
        class NoNeighborhood(Stencil):
            def kernel(self):
                pass

        with self.assertRaises(StencilException) as context:
            NoNeighborhood()

        self.assertTrue("neighborhood not properly" in context.exception.message)

    def test_bad_neighborhood_id(self):
        class OneNeighborhood(Stencil):
            Stencil.set_neighbor_definition([[(0, 0), (-1, 0)]])

            def kernel(self):
                pass

        one_neighborhood = OneNeighborhood()
        point = (0, 1)
        neighborhood_id = 2

        with self.assertRaises(StencilException) as context:
            one_neighborhood.neighbors(point, neighborhood_id).next()

        self.assertTrue("Undefined neighborhood identifier" in context.exception.message)




