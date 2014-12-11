from copy import deepcopy
import ctree

from ctree.c.nodes import *
from ctypes import c_int
from ctree.visitors import NodeTransformer
from stencil_code.stencil_model import *
from stencil_code.stencil_exception import StencilException


class StencilBackend(NodeTransformer):
    def __init__(self, input_grids=None, output_grid=None, kernel=None, arg_cfg=None,
                 fusable_nodes=None, testing=False):
        try:
            dir(self).index("visit_InteriorPointsLoop")
        except ValueError:
            raise StencilException("Error: {} must define a visit_InteriorPointsLoop method".format(type(self)))


        self.input_grids = input_grids
        self.output_grid = output_grid
        self.kernel = kernel
        self.ghost_depth = kernel.ghost_depth
        self.is_clamped = kernel.is_clamped
        self.is_copied = kernel.is_copied
        self.next_fresh_var = 0
        self.output_index = None
        self.neighbor_grid_name = None
        self.kernel_target = None
        self.offset_list = None
        self.var_list = []
        self.input_dict = {}
        self.input_names = []
        self.constants = kernel.constants
        self.distance = kernel.distance
        self.arg_cfg = arg_cfg
        self.fusable_nodes = fusable_nodes
        self.testing = testing
        super(StencilBackend, self).__init__()

    def visit_FunctionDecl(self, node):
        # This function grabs the input and output grid names which are used to
        # generate the proper array macros.
        # TODO: There may be a better way to do this? i.e. done at
        # initialization.
        for index, arg in enumerate(node.params):
            if index < len(self.input_grids):
                self.input_dict[arg.name] = self.input_grids[index]
                self.input_names.append(arg.name)
            else:
                self.output_grid_name = arg.name
        node.defn = list(map(self.visit, node.defn))
        node.name = "stencil_kernel"

    def gen_fresh_var(self):
        self.next_fresh_var += 1
        return "x%d" % self.next_fresh_var

    # def visit_InteriorPointsLoop(self, node):
    #     """
    #     must be implemented by subclass this is checked in __init__
    #     :param node:
    #     :return:
    #     """
    #     pass

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
        # TODO: unrolling blows up when neighborhood size id large. 27pt laplacian has 5K neighbors
        neighbors_id = node.neighbor_id
        # grid_name = node.grid_name
        # grid = self.input_dict[grid_name]
        zero_point = tuple([0 for x in range(self.kernel.dim)])
        self.neighbor_target = node.neighbor_target
        # self.neighbor_grid_name = grid_name
        body = []
        for x in self.kernel.neighbors(zero_point, neighbors_id):
            self.offset_list = list(x)
            for statement in node.body:
                body.append(self.visit(deepcopy(statement)))
        self.neighbor_target = None
        return body

    # Handle array references
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

    def visit_MathFunction(self, node):
        if str(node.func) == 'distance':
            zero_point = tuple([0 for _ in range(len(self.offset_list))])
            return Constant(int(self.distance(zero_point, self.offset_list)))
        elif str(node.func) == 'int':
            return Cast(c_int(), self.visit(node.args[0]))

    def gen_array_macro(self, arg, point):
        name = "_%s_array_macro" % arg
        return FunctionCall(SymbolRef(name), point)

    def visit_SymbolRef(self, node):  # pragma no cover
        if node.name in self.constants.keys():
            return Constant(self.constants[node.name])
        return node
