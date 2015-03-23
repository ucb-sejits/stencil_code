from _ast import FunctionDef, For, Attribute, Name
import unittest

import ast
from textwrap import dedent

import ctree.c.nodes as cn
from ctree.c.nodes import SymbolRef, Constant

from stencil_code.stencil_kernel import Stencil
from stencil_code.neighborhood import Neighborhood
from stencil_code.python_frontend import PythonToStencilModel
from stencil_code.stencil_exception import StencilException

import stencil_code.stencil_model as sm


class TestStencil(Stencil):
    neighborhoods = [Neighborhood.moore_neighborhood(radius=1, dim=2)]

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            for n in self.neighbors(x, 0):
                out_grid[x] += in_grid[n]


class TestStencilSelfRef(Stencil):
    neighborhoods = [Neighborhood.moore_neighborhood(radius=1, dim=2)]

    def __init__(self, backend='c', neighborhoods=None, boundary_handling='copy', **kwargs):
        self.weight = 0.555
        super(TestStencilSelfRef, self).__init__(backend, neighborhoods, boundary_handling, **kwargs)
    
    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            for n in self.neighbors(x, 0):
                out_grid[x] += self.weight * in_grid[n] * self.distance(n, x)


class TestStencilFrontend(unittest.TestCase):
    def setUp(self):
        self.transformer = PythonToStencilModel()

    def test_python_to_stencil_model(self):
        not_kernel = ast.parse("def not_kernel(self):\n    return 10")

        self.assertRaises(StencilException, self.transformer.visit, not_kernel)

    def test_function_def(self):
        """
        function def renames arguments to generated name
        """
        self.assertRaises(StencilException, self.transformer.visit, FunctionDef(name='bob'))

        kernel_def = ast.parse(dedent("""
        def kernel(self, in_grid, out_grid):
            pass
        """))
        node = self.transformer.visit(kernel_def)
        self.assertIsInstance(node, cn.CFile)
        decl = node.body[0]
        self.assertEqual(len(decl.params), 2)
        self.assertTrue(decl.params[0].name, "name_")
        self.assertTrue(decl.params[1].name, "name_")

        # NOTE: transformer modifies tree in place, cannot be re-run on same tree
        # NOTE2: names assigned will change as further invocations of transformer take place
        kernel_def = ast.parse(dedent("""
        def kernel(self, in_grid, out_grid):
            pass
        """))

        node = self.transformer.visit(kernel_def)
        self.assertIsInstance(node, cn.CFile)
        decl = node.body[0]
        self.assertEqual(len(decl.params), 2)
        self.assertTrue(decl.params[0].name, "name_")
        self.assertTrue(decl.params[1].name, "name_")

    def test_for(self):
        """
        if for iterates over interior_points or neighbors a semantic node is returned
        representing that instead of the usual for node
        """
        self.transformer.arg_name_map["a"] = "name_0"
        for_statement = ast.parse(dedent("""
        for x in self.interior_points(a):
            b += a[x]
        """))

        node = self.transformer.visit(for_statement)
        self.assertIsInstance(node.body[0], sm.InteriorPointsLoop)

        for_statement = ast.parse(dedent("""
        for x in self.neighbors(a, 0):
            b += a[x]
        """))

        node = self.transformer.visit(for_statement)
        self.assertIsInstance(node.body[0], sm.NeighborPointsLoop)

        for_statement = ast.parse(dedent("""
        for x in range(a):
            b += a[x]
        """))

        node = self.transformer.visit(for_statement)
        self.assertNotIsInstance(node.body[0], For)

    def test_call(self):
        """
        project currently supports two special calls
        distance which allows reference to function defined in the Stencil subclass
        int which allows casting
        Both are handed by replacing node with a MathFunction node
        :return:
        """
        call_statement = ast.parse(dedent("""
        self.distance(a, b)
        """))

        node = self.transformer.visit(call_statement)
        self.assertIsInstance(node.body[0], sm.MathFunction)

        call_statement = ast.parse(dedent("""
        int(b)
        """))

        node = self.transformer.visit(call_statement)
        self.assertIsInstance(node.body[0], sm.MathFunction)

        call_statement = ast.parse(dedent("""
        blobulate(x)
        """))

        node = self.transformer.visit(call_statement)
        self.assertIsInstance(node.body[0], cn.FunctionCall)

    def test_subscript(self):
        """
        subscripted variables that are parameters to the kernel function
        should be converted to GridElements, this will allow subsequent transforms
        to get the right n-d -> 1-d conversions and detect subscripts that are part
        of interior points or neighbor iterators
        :return:
        """
        subscript_statement = ast.parse(dedent("""
        a[x]
        """))
        self.transformer.arg_name_map['a'] = 'name_0'
        node = self.transformer.visit(subscript_statement)
        self.assertIsInstance(node.body[0], sm.GridElement)

        subscript_statement = ast.parse(dedent("""
        a[:-1]
        """))
        self.transformer.arg_name_map['a'] = 'name_0'
        self.assertRaises(StencilException, self.transformer.visit, subscript_statement)

    def test_attribute(self):
        # this stencil has an instance variable "weight"
        # this should be converted to a constant
        stencil = TestStencilSelfRef()
        transformer = PythonToStencilModel(stencil)
        
        node = Attribute(SymbolRef("self"), "weight", None)
        new_node = transformer.visit(node)
        self.assertIsInstance(new_node, Constant)

        # should not alter self.distance, that is handled magically at Call Node
        # node = ast.parse("self.distance(x)")
        node = Attribute(Name("self", None), "distance", None)

        new_node = transformer.visit(node)
        self.assertNotIsInstance(new_node, Constant)