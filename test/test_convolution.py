from stencil_code.library.convolution import ConvolutionFilter

import unittest
import numpy as np
from scipy.ndimage.filters import convolve


class TestConvolution(unittest.TestCase):
    def _check(self, actual, expected):
        np.testing.assert_allclose(actual, expected, atol=1e-3)

    def test_simple(self):
        a = (np.random.rand(256, 256) * 256).astype(np.float32)
        weights = np.random.rand(3, 3).astype(np.float32)
        specialized_convolve = ConvolutionFilter(convolution_array=weights,
                                                 backend='ocl')
        actual = specialized_convolve(a)
        expected = convolve(a, weights)
        self._check(actual[1:-1, 1:-1], expected[1:-1, 1:-1])
