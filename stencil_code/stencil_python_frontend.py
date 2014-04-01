
import ast

from ctree.visitors import NodeTransformer
import stencil_model as StencilModel


class PythonToStencilModel(NodeTransformer):
    def visit_For(self, node):
        node.body = list(map(self.visit, node.body))
        if type(node.iter) is ast.Call and \
           type(node.iter.func) is ast.Attribute:
            if node.iter.func.attr is 'interior_points':
                ret_node = StencilModel.InteriorPointsLoop(node)
                ret_node.base_node = node
                return ret_node
            elif node.iter.func.attr is 'neighbors':
                ret_node = StencilModel.NeighborPointsLoop(node)
                ret_node.base_node = node
                return ret_node
        return node

    def visit_FunctionCall(self, node):
        node.args = list(map(self.visit, node.args))
        if str(node.func) == 'distance' or str(node.func) == 'int':
            ret_node = StencilModel.MathFunction()
            ret_node.base_node = node
            return ret_node

        return node
