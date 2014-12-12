import unittest

from stencil_code.stencil_kernel import Stencil
from stencil_code.backend.omp import *
from stencil_code.stencil_exception import StencilException
from ctree.transformations import PyBasicConversions


class TestStencilBackend(unittest.TestCase):
    def test_bad_impl(self):
        class NoInteriorPoints(StencilBackend):
            pass

        with self.assertRaises(StencilException) as context:
            NoInteriorPoints()

        self.assertTrue("must define a visit_InteriorPointsLoop method" in context.exception.args[0])
