
import ast

from ctree.transformations import PyBasicConversions
from stencil_model import*


class PythonToStencilModel(PyBasicConversions):
    # Strip self paramater from kernel
    def visit_FunctionDef(self, node):
        if node.name == 'kernel':
            node.args.args = node.args.args[1:]
        return super(PythonToStencilModel, self).visit_FunctionDef(node)

    def visit_For(self, node):
        node.body = list(map(self.visit, node.body))
        if type(node.iter) is ast.Call and \
           type(node.iter.func) is ast.Attribute:
            if node.iter.func.attr is 'interior_points':
                return InteriorPointsLoop(target=node.target.id,
                                          body=node.body)
            elif node.iter.func.attr is 'neighbors':
                return NeighborPointsLoop(
                    neighbor_id=node.iter.args[1].n,
                    grid_name=node.iter.func.value.id,
                    neighbor_target=node.target.id,
                    body=node.body
                )
        return super(PythonToStencilModel, self).visit_For(node)

    def visit_Call(self, node):
        node = super(PythonToStencilModel, self).visit_Call(node)
        if str(node.func) == 'distance' or str(node.func) == 'int':
            node.args = list(map(self.visit, node.args))
            return MathFunction(func=node.func, args=node.args)
        return node

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        slice = self.visit(node.slice)
        return GridElement(
            grid_name=value.name,
            target=slice.value
        )
