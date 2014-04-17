import unittest

from stencil_code.stencil_kernel import StencilKernel


class TestStencilKernel(unittest.TestCase):
    def test_no_kernel_exception(self):
        class BadKernel(StencilKernel):
            pass
        self.assertRaises(Exception, BadKernel)


