
import ast

from ctree.visitors import NodeTransformer
import stencil_model as StencilModel


class PythonToStencilModel(NodeTransformer):
    def visit_For(self, node):
        node.body = list(map(self.visit, node.body))
        if type(node.iter) is ast.Call and \
           type(node.iter.func) is ast.Attribute:
            if node.iter.func.attr is 'interior_points':
                return StencilModel.InteriorPointsLoop(node)
            elif node.iter.func.attr is 'neighbors':
                return StencilModel.NeighborPointsLoop(node)
        return node

    def visit_FunctionCall(self, node):
        node.args = list(map(self.visit, node.args))
        if str(node.func) == 'distance' or str(node.func) == 'int':
            return StencilModel.MathFunction(node)
        return node
