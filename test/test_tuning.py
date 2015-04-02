import unittest
from stencil_code.library.diagnostic_stencil import DiagnosticStencil
from stencil_code.library.jacobi_stencil import Jacobi

from stencil_code.stencil_kernel import Stencil, SpecializedStencil

import numpy as np


class TestTuning(unittest.TestCase):

    def test_opentuner_reset(self):
        specialized_stencil = SpecializedStencil(DiagnosticStencil(), 'ocl')
        input_grid = np.ones([100, 100])
        specialized_stencil.args_to_subconfig([input_grid])
        for _ in range(50):
            tuner_config = next(specialized_stencil._tuner.configs)
            specialized_stencil._tuner.report(time=10.0)

        input_grid = np.ones([256, 256])
        specialized_stencil.args_to_subconfig([input_grid])
        for _ in range(50):
            tuner_config = next(specialized_stencil._tuner.configs)
            specialized_stencil._tuner.report(time=10.0)
        #     print tuner_config

    def test_jacobi_reset(self):
        rows, cols = 64, 64
        in_img = np.ones([rows, cols]).astype(np.float32)
        stencil = Jacobi(backend="ocl", boundary_handling="clamp")

        for _ in range(7):
            out_img = stencil(in_img)

        in_img2 = np.ones([100,200]).astype(np.float32)
        for _ in range(7):
            out_img = stencil(in_img2)
