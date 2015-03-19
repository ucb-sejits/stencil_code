import unittest
from opentuner.resultsdb.models import Result
from stencil_code.library.diagnostic_stencil import DiagnosticStencil
from stencil_code.library.jacobi_stencil import Jacobi

from stencil_code.stencil_exception import StencilException
from stencil_code.stencil_kernel import Stencil, SpecializedStencil

import numpy as np


class TestTuning(unittest.TestCase):

    def test_opentuner_reset(self):
        specialized_stencil = SpecializedStencil(DiagnosticStencil(), 'ocl')
        input_grid = np.ones([100, 100])
        # self.assertEquals(input_grid.shape, (100, 100))
        specialized_stencil.args_to_subconfig([input_grid])
        tuner_config = next(specialized_stencil._tuner.configs)
        specialized_stencil._tuner.report(time=10.0)

        print tuner_config
        # input_grid = np.ones([256, 256])
        # specialized_stencil.args_to_subconfig([input_grid])
        # tuner_config = next(specialized_stencil._tuner.configs)
        # print tuner_config
        # how to test to see if changes if change input size?




