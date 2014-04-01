import ast
from ctree.dotgen import DotGenVisitor


class StencilModelNode(ast.AST):
    _fields = ['base_node']

    def __init__(self, base_node=None):
        self.base_node = base_node
        super(StencilModelNode, self).__init__()

    def _to_dot(self):
        return StencilModelDotGen.visit(self)


class InteriorPointsLoop(StencilModelNode):
    pass


class NeighborPointsLoop(StencilModelNode):
    pass


class MathFunction(StencilModelNode):
    pass


class StencilModelDotGen(DotGenVisitor):
    def label_InteriorPointsLoop(self, node):
        return r"%s" % "InteriorPointsLoop"

    def label_NeighborPointsLoop(self, node):
        return r"%s" % "NeighborPointsLoop"

    def label_MathFunction(self, node):
        return r"%s" % "MathFunction"
