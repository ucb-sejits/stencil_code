
import ast
from _ast import Name
from _ast import Index
from numbers import Number
from ctree.transformations import PyBasicConversions
from ctree.c.nodes import SymbolRef, Constant
from .stencil_model import GridElement, InteriorPointsLoop, NeighborPointsLoop, \
    MathFunction, MultiPointsLoop
from .stencil_exception import StencilException

import sys


class PythonToStencilModel(PyBasicConversions):
    def __init__(self, stencil_kernel=None, arg_names=None):
        super(PythonToStencilModel, self).__init__()
        self.stencil_kernel = stencil_kernel
        self.arg_names = arg_names
        self.arg_name_map = {}

    # noinspection PyPep8Naming
    def visit_FunctionDef(self, node):
        """
        strip self parameter from the kernel
        :param node:
        :return:
        """
        if node.name == 'kernel':
            node.args.args = node.args.args[1:]
        else:
            raise StencilException("AST, FunctionDef '{}' found, should only be 'kernel'".format(node.name))

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
                name = SymbolRef.unique().name
                if sys.version_info >= (3, 0):
                    self.arg_name_map[arg.arg] = name
                    arg.arg = name
                else:
                    self.arg_name_map[arg.id] = name
                    arg.id = name
        return super(PythonToStencilModel, self).visit_FunctionDef(node)

    # noinspection PyPep8Naming
    def visit_For(self, node):
        node.body = list(map(self.visit, node.body))
        if type(node.iter) is ast.Call and \
           type(node.iter.func) is ast.Attribute:
            if node.iter.func.attr is 'interior_points':
                return InteriorPointsLoop(target=node.target.id,
                                          body=node.body,
                                          grid_name=self.arg_name_map[node.iter.args[0].id])
            elif node.iter.func.attr is 'neighbors':
                # neighbor method should have default neighbor id of 0
                neighbor_id = 0 if len(node.iter.args) < 2 else node.iter.args[1].n
                return NeighborPointsLoop(
                    reference_point=node.iter.args[0].id,
                    neighbor_id=neighbor_id,
                    neighbor_target=node.target.id,
                    body=node.body
                )
            elif node.iter.func.attr is 'multi_points':
                return MultiPointsLoop(
                    input_target=node.target.elts[0].id,
                    output_target=node.target.elts[1].id,
                    coefficient=node.target.elts[2].id,
                    reference_point=node.iter.args[0].id,
                    body=node.body
                )
        return super(PythonToStencilModel, self).visit_For(node)  # pragma no cover

    # noinspection PyPep8Naming
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

    # noinspection PyPep8Naming
    def visit_Subscript(self, node):
        """
        subscripts in stencil specializers must be simple Index class, essentially
        just a variable
        :param node:
        :return:
        """
        value = self.visit(node.value)
        if isinstance(node.slice, Index):
            sliced = self.visit(node.slice.value)
            if isinstance(sliced, str):
                sliced = SymbolRef(sliced)
            # if not hasattr(value, 'name'):
            #     array_name = value.name
            # else:
            #     array_name = self.arg_name_map[value.name]
            return GridElement(
                # grid_name=array_name,
                grid_name=self.arg_name_map[value.name],
                target=sliced
            )
        else:
            raise StencilException("{} subscript {} should be a Index not a {}".format(
                node.value.id, node.slice, node.slice.__class__.__name__))

    # noinspection PyPep8Naming
    def visit_Attribute(self, node):
        """
        This allows a reference to a numeric value of the stencil_kernel in
        the kernel function, other uses will most likely blow up
        :param node:
        :return:
        """
        if isinstance(node.value, Name):
            if node.value.id == 'self' and self.stencil_kernel is not None:
                value = getattr(self.stencil_kernel, node.attr)
                if isinstance(value, Number):
                    return Constant(value)
        if isinstance(node.value, SymbolRef):
            if node.value.name == 'self' and self.stencil_kernel is not None:
                value = getattr(self.stencil_kernel, node.attr)
                if isinstance(value, Number):
                    return Constant(value)

        return node
