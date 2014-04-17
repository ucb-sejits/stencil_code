from copy import deepcopy

from ctree.c.types import *
from ctree.c.nodes import *
from ctree.omp.nodes import *
from ctree.visitors import NodeTransformer
from ctree.templates.nodes import StringTemplate
from stencil_model import *
from ctree.cpp.nodes import CppDefine
from stencil_grid import *


class StencilOmpTransformer(NodeTransformer):
    def __init__(self, input_grids=None, output_grid=None, kernel=None):
        # TODO: Give these wrapper classes?
        if not input_grids and not output_grid and not kernel: #pragma: no # cover
            width = 50
            radius = 1
            output_grid = StencilGrid([width, width])
            output_grid.ghost_depth = radius
            in_grid = StencilGrid([width, width])
            in_grid.ghost_depth = radius

            for x in range(0, width):
                for y in range(0, width):
                    in_grid.data[(x, y)] = 1.0

            for x in range(-radius, radius+1):
                for y in range(-radius, radius+1):
                    in_grid.neighbor_definition[1].append((x, y))

            input_grids = [in_grid]
            kernel = {'constants': None, 'distance': None}
        self.input_grids = input_grids
        self.output_grid = output_grid
        self.ghost_depth = output_grid.ghost_depth
        self.next_fresh_var = 0
        self.output_index = None
        self.neighbor_grid_name = None
        self.kernel_target = None
        self.offset_list = None
        self.var_list = []
        self.input_dict = {}
        self.constants = kernel.constants
        self.distance = kernel.distance
        super(StencilOmpTransformer, self).__init__()

    def visit_FunctionDecl(self, node):
        # This function grabs the input and output grid names which are used to
        # generate the proper array macros.
        # TODO: There may be a better way to do this? i.e. done at
        # initialization.
        for index, arg in enumerate(node.params):
            if index < len(self.input_grids):
                self.input_dict[arg.name] = self.input_grids[index]
            else:
                self.output_grid_name = arg.name
        node.defn = list(map(self.visit, node.defn))
        node.name = "stencil_kernel"
        for index, arg in enumerate(self.input_grids + (self.output_grid,)):
            defname = "_%s_array_macro" % node.params[index].name
            params = ','.join(["_d"+str(x) for x in range(arg.dim)])
            params = "(%s)" % params
            calc = "((_d%d)" % (arg.dim - 1)
            for x in range(arg.dim - 1):
                dim = str(int(arg.data.strides[x]/arg.data.itemsize))
                calc += "+((_d%s) * %s)" % (str(x), dim)
            calc += ")"
            params = ["_d"+str(x) for x in range(arg.dim)]
            node.defn.insert(0, CppDefine(defname, params, calc))
        abs_decl = FunctionDecl(
            Int(), SymbolRef('abs'), [SymbolRef('n', Int())]
        )
        return [abs_decl, node]

    def gen_fresh_var(self):
        self.next_fresh_var += 1
        return "x%d" % self.next_fresh_var

    def visit_InteriorPointsLoop(self, node):
        dim = len(self.output_grid.shape)
        self.kernel_target = node.target
        curr_node = None
        ret_node = None
        for d in range(dim):
            target = SymbolRef(self.gen_fresh_var())
            self.var_list.append(target.name)
            for_loop = For(
                Assign(SymbolRef(target.name, Int()),
                       Constant(self.ghost_depth)),
                LtE(target,
                    Constant(
                        self.output_grid.shape[d] -
                        self.ghost_depth - 1)
                    ),
                PostInc(target),
                [])

            if d == 0:
                ret_node = for_loop
            else:
                curr_node.body = [for_loop]
                if d == dim - 2:
                    curr_node.body.insert(0, OmpParallelFor())
                elif d == dim - 1:
                    curr_node.body.insert(0, OmpIvDep())
            curr_node = for_loop
        self.output_index = self.gen_fresh_var()
        pt = [SymbolRef(x) for x in self.var_list]
        macro = self.gen_array_macro(self.output_grid_name, pt)
        curr_node.body = [Assign(SymbolRef(self.output_index, Int()),
                                 macro)]
        for elem in map(self.visit, node.body):
            if type(elem) == list:
                curr_node.body.extend(elem)
            else:
                curr_node.body.append(elem)
        self.kernel_target = None
        return ret_node

    def visit_NeighborPointsLoop(self, node):
        neighbors_id = node.neighbor_id
        grid_name = node.grid_name
        grid = self.input_dict[grid_name]
        zero_point = tuple([0 for x in range(grid.dim)])
        self.neighbor_target = node.neighbor_target
        self.neighbor_grid_name = grid_name
        body = []
        for x in grid.neighbors(zero_point, neighbors_id):
            self.offset_list = list(x)
            for statement in node.body:
                body.append(self.visit(deepcopy(statement)))
        self.neighbor_target = None
        return body

    # Handle array references
    def visit_GridElement(self, node):
        grid_name = node.grid_name
        target = node.target
        if isinstance(target, SymbolRef):
            target = target.name
            if target == self.kernel_target:
                if grid_name is self.output_grid_name:
                    return ArrayRef(SymbolRef(self.output_grid_name),
                                    SymbolRef(self.output_index))
                elif grid_name in self.input_dict:
                    # grid = self.input_dict[grid_name]
                    pt = list(map(lambda x: SymbolRef(x), self.var_list))
                    index = self.gen_array_macro(grid_name, pt)
                    return ArrayRef(SymbolRef(grid_name), index)
            elif grid_name == self.neighbor_grid_name:
                pt = list(map(lambda x, y: Add(SymbolRef(x), SymbolRef(y)),
                              self.var_list, self.offset_list))
                index = self.gen_array_macro(grid_name, pt)
                return ArrayRef(SymbolRef(grid_name), index)
        elif isinstance(target, FunctionCall) or \
                isinstance(target, MathFunction):
            return ArrayRef(SymbolRef(grid_name), self.visit(target))
        return node

    def visit_MathFunction(self, node):
        if str(node.func) == 'distance':
            zero_point = tuple([0 for _ in range(len(self.offset_list))])
            return Constant(int(self.distance(zero_point, self.offset_list)))
        elif str(node.func) == 'int':
            return Cast(Int(), self.visit(node.args[0]))

    def gen_array_macro(self, arg, point):
        name = "_%s_array_macro" % arg
        return FunctionCall(SymbolRef(name), point)

    def visit_SymbolRef(self, node):
        if node.name in self.constants.keys():
            return Constant(self.constants[node.name])
        return node
