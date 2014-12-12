
import ast
import ctree

from ctree.transformations import PyBasicConversions
from ctree.c.nodes import SymbolRef
from .stencil_model import GridElement, InteriorPointsLoop, NeighborPointsLoop, \
    MathFunction

import sys


class PythonToStencilModel(PyBasicConversions):
    def __init__(self, arg_names=None):
        super(PythonToStencilModel, self).__init__()
        self.arg_names = arg_names
        self.arg_name_map = {}

    # Strip self parameter from kernel
    def visit_FunctionDef(self, node):
        if node.name == 'kernel':
            node.args.args = node.args.args[1:]
        if self.arg_names is not None:  # pragma no cover
            for index, arg in enumerate(node.args.args):
                new_name = self.arg_names[index]
                if sys.version_info >= (3, 0):
                    self.arg_name_map[arg.arg] = new_name
                else:
                    self.arg_name_map[arg.id] = new_name
                arg.id = new_name
        else:
            for index, arg in enumerate(node.args.args):  # pragma no cover
                if sys.version_info >= (3, 0):
                    self.arg_name_map[arg.arg] = arg.arg
                else:
                    self.arg_name_map[arg.id] = arg.id
        return super(PythonToStencilModel, self).visit_FunctionDef(node)

    def visit_For(self, node):
        node.body = list(map(self.visit, node.body))
        if type(node.iter) is ast.Call and \
           type(node.iter.func) is ast.Attribute:
            if node.iter.func.attr is 'interior_points':
                return InteriorPointsLoop(target=node.target.id,
                                          body=node.body)
            elif node.iter.func.attr is 'neighbors':
                # neighbor method should have default neighbor id of 0
                neighbor_id = 0 if len(node.iter.args) < 2 else node.iter.args[1].n
                return NeighborPointsLoop(
                    neighbor_id=neighbor_id,
                    neighbor_target=node.target.id,
                    body=node.body
                )
        return super(PythonToStencilModel, self).visit_For(node)  # pragma no cover

    def visit_Call(self, node):
        node = super(PythonToStencilModel, self).visit_Call(node)

        func_name = None
        if isinstance(node.func, SymbolRef):
            func_name = node.func.name
        else:
            try:
                func_name = node.func.attr
            except Exception:  # pragma no cover
                pass

        if func_name and func_name == 'distance' or func_name == 'int':
            node.args = list(map(self.visit, node.args))
            return MathFunction(func=func_name, args=node.args)
        return node

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        slice = self.visit(node.slice)
        return GridElement(
            grid_name=self.arg_name_map[value.name],
            target=slice.value
        )
