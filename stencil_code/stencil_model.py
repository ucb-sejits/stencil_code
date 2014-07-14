import ast
from ctree.dotgen import DotGenVisitor
from ctree.templates.nodes import StringTemplate
from ctree.c.nodes import *
from ctree.ocl.macros import *
import ctypes as ct


class StencilModelNode(ast.AST):
    def _to_dot(self):  # pragma: no cover
        return StencilModelDotGen.visit(self)

    def label(self):
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

    def label(self):
        return r"%s" % (self.grid_name)


class MacroDefns(StencilModelNode):
    _fields = ['body']

    def __init__(self, body=None):
        self.body = body
        super(MacroDefns, self).__init__()

    def add_undef(self):
        for macro in self.body[:]:
            self.body.insert(0, StringTemplate("#undef %s" % macro.name))

    def __str__(self):
        return "\n".join(map(str, self.body))

class OclNeighborLoop(StencilModelNode):
    # _fields = ['body']

    def __init__(self, body=None, out_grid_shape=None, ghost_depth=None):
        self.body = body
        self.out_grid_shape = out_grid_shape
        self.ghost_depth = ghost_depth
        super(OclNeighborLoop, self).__init__()

    def __str__(self):
        return "\n".join(map(str, self.body))

    def set_global_index(self, padding):
        dim = len(self.out_grid_shape)
        index = Add(get_global_id(dim - 1), Constant(self.ghost_depth * padding))
        for d in reversed(range(dim - 1)):
            index = Add(
                Mul(
                    index,
                    Constant(self.out_grid_shape[d])
                ),
                Add(get_global_id(d), Constant(self.ghost_depth * padding))
            )
        self.body.insert(0, Assign(SymbolRef('global_index', ct.c_int()),
            index))

class LoadSharedMemBlock(StencilModelNode):
    def __init__(self, decls=None, body=None):
        self.decls = decls
        self.body = body
        super(LoadSharedMemBlock, self).__init__()

    def __str__(self):
        s = "\n".join(map(str, self.decls))
        s += "\n"
        s = "\n".join(map(str, self.body))
        return s

    def remove_types_from_decl(self):
        for item in self.decls:
            item.left.type = None

    def set_block_idx(self, name):
        self.body[0].body.insert(0, StringTemplate("global_index = tid;"))


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
