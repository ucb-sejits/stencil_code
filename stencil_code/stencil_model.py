import ast
from ctree.dotgen import DotGenVisitor
from ctree.templates.nodes import StringTemplate
from ctree.c.nodes import *
from ctree.ocl.macros import *
import ctypes as ct


class StencilModelNode(ast.AST):
    def _to_dot(self):  # pragma: no cover
        return StencilModelDotGen.visit(self)

    def label(self): # pragma: no cover
        return r"%s" % ""


class InteriorPointsLoop(StencilModelNode):
    _fields = ['target', 'body']

    def __init__(self, target=None, body=None):
        self.target = target
        self.body = body
        super(InteriorPointsLoop, self).__init__()


class NeighborPointsLoop(StencilModelNode):
    _fields = ['neighbor_id', 'grid_name',
               'neighbor_target', 'body']

    def __init__(self, neighbor_id=None, grid_name=None,
                 neighbor_target=None, body=None):
        self.neighbor_id = neighbor_id
        self.grid_name = grid_name
        self.neighbor_target = neighbor_target
        self.body = body
        super(NeighborPointsLoop, self).__init__()


class MathFunction(StencilModelNode):
    _fields = ['func', 'args']

    def __init__(self, func=None, args=None):
        self.func = func
        self.args = args
        super(MathFunction, self).__init__()


class GridElement(StencilModelNode):
    _fields = ['grid_name', 'target']

    def __init__(self, grid_name=None, target=None):
        self.grid_name = grid_name
        self.target = target
        super(StencilModelNode, self).__init__()

    def label(self):  # pragma no cover
        return r"%s" % (self.grid_name)


class StencilModelDotGen(DotGenVisitor):  # pragma: no cover
    def label_InteriorPointsLoop(self, node):
        return r"%s" % "InteriorPointsLoop"

    def label_NeighborPointsLoop(self, node):
        return r"%s" % "NeighborPointsLoop"

    def label_MathFunction(self, node):
        return r"%s" % "MathFunction"

    def label_GridElement(self, node):
        return r"%s" % "GridElement"

    def label_MacroDefns(self, node):
        return r"MacroDefns"

    def label_LoadSharedMemBlock(self, node):
        return r"LoadSharedMemBlock"
