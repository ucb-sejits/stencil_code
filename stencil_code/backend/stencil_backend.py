from copy import deepcopy
import numpy as np

from ctypes import c_int
from ctree.visitors import NodeTransformer
from stencil_code.stencil_model import *
from stencil_code.stencil_exception import StencilException


class StencilBackend(NodeTransformer):
    def __init__(self, parent_lazy_specializer=None, arg_cfg=None,
                 fusable_nodes=None, testing=False):
        try:
            dir(self).index("visit_InteriorPointsLoop")
        except ValueError:
            raise StencilException("Error: {} must define a visit_InteriorPointsLoop method".format(type(self)))

        self.input_grids = arg_cfg
        self.output_grid_name = None
        self.parent_lazy_specializer = parent_lazy_specializer
        self.ghost_depth = parent_lazy_specializer.ghost_depth
        self.is_clamped = parent_lazy_specializer.is_clamped
        self.is_copied = parent_lazy_specializer.is_copied
        self.next_fresh_var = 0
        self.output_index = None
        self.neighbor_grid_name = None
        self.neighbor_target = None
        self.kernel_target = None
        self.offset_list = []
        self.offset_dict = {}
        self.var_list = []
        self.input_dict = {}
        self.input_names = []
        self.index_target_dict = {}
        self.constants = parent_lazy_specializer.constants
        self.distance = parent_lazy_specializer.distance
        self.arg_cfg = arg_cfg
        self.fusable_nodes = fusable_nodes
        self.testing = testing
        super(StencilBackend, self).__init__()

    def function_decl_helper(self, node):
        # This function grabs the input and output grid names which are used to
        # generate the proper array macros.

        if len(self.arg_cfg) == len(node.params) - 1:
            # typically passed in arguments will not include output, in which case
            # it is coerced to be the same type as the first argument
            self.arg_cfg += (self.arg_cfg[0],)

        for index, arg in enumerate(self.arg_cfg):
            # fix up type of parameters, build a dictionary mapping param name to argument info
            param = node.params[index]
            param.type = np.ctypeslib.ndpointer(arg.dtype, arg.ndim, arg.shape)()
            self.input_dict[param.name] = arg
            self.input_names.append(param.name)
        self.output_grid_name = node.params[-1].name

        node.defn = list(map(self.visit, node.defn))
        node.name = "stencil_kernel"

    def gen_fresh_var(self):
        self.next_fresh_var += 1
        return "x%d" % self.next_fresh_var

    # noinspection PyPep8Naming
    def visit_NeighborPointsLoop(self, node):
        """
        unrolls the neighbor points loop, appending each current block of the body to a new
        body for each neighbor point, a side effect of this is local python functions of the
        neighbor point can be collapsed out, for example, a custom python distance function based
        on neighbor distance can be resolved at transform time
        DANGER: this blows up on large neighborhoods
        :param node:
        :return:
        """
        # TODO: unrolling blows up when neighborhood size is large.
        neighbors_id = node.neighbor_id
        zero_point = tuple([0 for x in range(self.parent_lazy_specializer.dim)])
        self.neighbor_target = node.neighbor_target

        self.index_target_dict[node.neighbor_target] = self.index_target_dict[node.reference_point]
        body = []
        for x in self.parent_lazy_specializer.neighbors(zero_point, neighbors_id):
            # TODO: add line below to manage indices that refer to neighbor points loop
            self.offset_list = list(x)
            self.offset_dict[self.neighbor_target] = list(x)
            for statement in node.body:
                body.append(self.visit(deepcopy(statement)))
        self.index_target_dict.pop(node.neighbor_target, None)
        return body

    # noinspection PyPep8Naming
    def visit_GridElement(self, node):  # pragma no cover
        grid_name = node.grid_name
        target = node.target
        if isinstance(target, SymbolRef):
            target = target.name
            if target == self.kernel_target:
                if grid_name is self.output_grid_name:
                    return ArrayRef(SymbolRef(self.output_grid_name),
                                    self.output_index)
                elif grid_name in self.input_dict:
                    # grid = self.input_dict[grid_name]
                    pt = list(map(lambda x: SymbolRef(x), self.var_list))
                    index = self.gen_array_macro(grid_name, pt)
                    return ArrayRef(SymbolRef(grid_name), index)
            elif grid_name == self.neighbor_grid_name:
                pt = list(map(lambda x, y: Add(SymbolRef(x), Constant(y)),
                              self.var_list, self.offset_list))
                index = self.gen_array_macro(grid_name, pt)
                return ArrayRef(SymbolRef(grid_name), index)
        elif isinstance(target, FunctionCall) or \
                isinstance(target, MathFunction):
            return ArrayRef(SymbolRef(grid_name), self.visit(target))
        raise Exception("Found GridElement that is not supported")

    # noinspection PyPep8Naming
    def visit_MathFunction(self, node):
        if str(node.func) == 'distance':
            zero_point = tuple([0 for _ in range(len(self.offset_list))])
            return Constant(self.distance(zero_point, self.offset_list))
        elif str(node.func) == 'int':
            return Cast(c_int(), self.visit(node.args[0]))

    def gen_array_macro(self, arg, point):
        name = "_%s_array_macro" % arg
        return FunctionCall(SymbolRef(name), point)

    # noinspection PyPep8Naming
    def visit_SymbolRef(self, node):  # pragma no cover
        if node.name in self.constants.keys():
            return Constant(self.constants[node.name])
        return node
