import unittest

import numpy as np
from textwrap import dedent

import ctree.c.nodes as ctree_nodes

import stencil_code.stencil_model as stencil_model
from stencil_code.library.diagnostic_stencil import DiagnosticStencil
from stencil_code.backend.stencil_backend import StencilBackend

from stencil_code.stencil_kernel import Stencil, StencilArgConfig
from stencil_code.backend.omp import *
from stencil_code.stencil_exception import StencilException


class TestStencilBackend(unittest.TestCase):
    """
    StencilBackend will be called after the PythonToStencilModel transform so it
    expects:
    self removed from kernel parameters
    other kernel parameters have been renamed
    """

    def setUp(self):
        self.diagnostic_stencil = DiagnosticStencil(backend='c')
        self.simple_arg_config = [StencilArgConfig(5, np.float32, 2, (5, 5))]

    def test_function_decl_helper(self):
        """
        The function_decl helper
        parameters in tree are initially untyped, but are given types based on FunctionDecl node
        arg_cfg, initially does not include the output so it grows by 1

        :return:
        """
        stencil_backend = StencilBackend(
            parent_lazy_specializer=self.diagnostic_stencil,
            arg_cfg=self.simple_arg_config
        )

        shape = (5, 5)
        input_grid = np.ones(shape, dtype=np.float32)
        function_decl = FunctionDecl(
            return_type=None, name='kernel',
            params=[SymbolRef('name_0'), SymbolRef('name_1')],
            defn=None,
        )

        self.assertIsNone(function_decl.params[0].type)
        self.assertIsNone(function_decl.params[1].type)
        self.assertEqual(len(stencil_backend.arg_cfg), 1)
        stencil_backend.function_decl_helper(function_decl)
        self.assertIsNotNone(function_decl.params[0].type)
        self.assertIsNotNone(function_decl.params[1].type)
        self.assertEqual(len(stencil_backend.arg_cfg), 2)

        output_grid = self.diagnostic_stencil(input_grid)

        self.assertIsNotNone(output_grid)

    def test_interior_points_loop(self):
        stencil_backend = StencilBackend(
            parent_lazy_specializer=self.diagnostic_stencil,
            arg_cfg=self.simple_arg_config
        )

        shape = (5, 5)
        input_grid = np.ones(shape, dtype=np.float32)

        kernel_def = ast.parse(dedent("""
        Iabc
        """))
        interior_points_node = stencil_model.InteriorPointsLoop(
            target="name0", body=[], grid_name="name_0"
        )
        node = stencil_backend.visit(interior_points_node)

        self.assertIsNotNone(node)