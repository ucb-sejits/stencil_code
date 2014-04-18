import ast
from ctree.dotgen import DotGenVisitor


class StencilModelNode(ast.AST):
    def _to_dot(self):  # pragma: no cover
        return StencilModelDotGen.visit(self)


class InteriorPointsLoop(StencilModelNode):
    _fields = ['target', 'body']

    def __init__(self, target=None, body=None):
        self.target = target
        self.body = body


class NeighborPointsLoop(StencilModelNode):
    _fields = ['neighbor_id', 'grid_name',
               'neighbor_target', 'body']

    def __init__(self, neighbor_id=None, grid_name=None,
                 neighbor_target=None, body=None):
        self.neighbor_id = neighbor_id
        self.grid_name = grid_name
        self.neighbor_target = neighbor_target
        self.body = body


class MathFunction(StencilModelNode):
    _fields = ['func', 'args']

    def __init__(self, func=None, args=None):
        self.func = func
        self.args = args


class GridElement(StencilModelNode):
    _fields = ['grid_name', 'target']

    def __init__(self, grid_name=None, target=None):
        self.grid_name = grid_name
        self.target = target


class StencilModelDotGen(DotGenVisitor):  # pragma: no cover
    def label_InteriorPointsLoop(self, node):
        return r"%s" % "InteriorPointsLoop"

    def label_NeighborPointsLoop(self, node):
        return r"%s" % "NeighborPointsLoop"

    def label_MathFunction(self, node):
        return r"%s" % "MathFunction"

    def label_GridElement(self, node):
        return r"%s" % "GridElement"
