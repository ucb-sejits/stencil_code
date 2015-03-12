import unittest

import ast
import numpy as np

from stencil_code.stencil_kernel import Stencil
from stencil_code.neighborhood import Neighborhood
from stencil_code.python_frontend import PythonToStencilModel
from stencil_code.stencil_exception import StencilException


class LookupStencil(Stencil):
    neighborhoods = [Neighborhood.moore_neighborhood(radius=1, dim=2)]

    lut = np.array([[
        0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 0,
    ]])

    def kernel(self, in_grid, lut, out_grid):
        for x in self.interior_points(out_grid):
            acc = 0
            for n in self.neighbors(x, 0):
                acc += in_grid[n]
            out_grid[x] = lut[acc]


class TestStencilFrontend(unittest.TestCase):

    def test_python_to_stencil_model(self):
        transform = PythonToStencilModel()

        not_kernel = ast.parse("def not_kernel(self):\n    return 10")

        self.assertRaises(StencilException, transform.visit, not_kernel)

    def test_parameter_assignment(self):
        import logging
        logging.basicConfig(level=20)

        ls = LookupStencil(backend='c')

        start_grid = np.random.randint(2, size=(10, 10))

        end_grid = ls(start_grid, ls.lut)

        print(start_grid)

        print('='*80)

        print(end_grid)


