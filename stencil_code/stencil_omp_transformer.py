import ast
from copy import deepcopy

from ctree.c.types import *
from ctree.c.nodes import *
from ctree.omp.nodes import *
from ctree.visitors import NodeTransformer
from ctree.transformations import PyBasicConversions
from stencil_model import *


class StencilOmpTransformer(NodeTransformer):
    def __init__(self, input_grids, output_grid, kernel):
        # TODO: Give these wrapper classes?
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
        for index, arg in enumerate(node.params[1:]):
            if index < len(self.input_grids):
                self.input_dict[arg.name] = self.input_grids[index]
            else:
                self.output_grid_name = arg.name
        node.defn = list(map(self.visit, node.defn))
        return node

    def gen_fresh_var(self):
        self.next_fresh_var += 1
        return "x%d" % self.next_fresh_var

    def visit_InteriorPointsLoop(self, node):
        node = node.base_node
        dim = len(self.output_grid.shape)
        self.kernel_target = node.target.id
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
        node = node.base_node
        neighbors_id = node.iter.args[1].n
        grid_name = node.iter.func.value.id
        grid = self.input_dict[grid_name]
        zero_point = tuple([0 for x in range(grid.dim)])
        self.neighbor_target = node.target.id
        self.neighbor_grid_name = grid_name
        body = []
        statement = node.body[0]
        for x in grid.neighbors(zero_point, neighbors_id):
            self.offset_list = list(x)
            for statement in node.body:
                body.append(self.visit(deepcopy(statement)))
        self.neighbor_target = None
        return body

    # Handle array references
    def visit_Subscript(self, node):
        grid_name = node.value.name
        target = node.slice.value
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
        node = node.base_node
        if str(node.func) == 'distance':
            zero_point = tuple([0 for _ in range(len(self.offset_list))])
            return Constant(int(self.distance(zero_point, self.offset_list)))
        elif str(node.func) == 'int':
            return Cast(Int(), self.visit(node.args[0]))
        print(node.func)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            zero_point = tuple([0 for _ in range(len(self.offset_list))])
            return Constant(self.distance(zero_point, self.offset_list))
        if node.func.id == 'distance':
            zero_point = tuple([0 for _ in range(len(self.offset_list))])
            return Constant(int(self.distance(zero_point, self.offset_list)))
        elif node.func.id == 'int':
            return Cast(Int(), self.visit(node.args[0]))
        node.args = list(map(self.visit, node.args))
        return node

    def gen_array_macro(self, arg, point):
        name = "_%s_array_macro" % arg
        return FunctionCall(SymbolRef(name), point)

    def visit_AugAssign(self, node):
        # TODO: Handle all types?
        value = self.visit(node.value)
        # HACK to get this to work, PyBasicConversions will skip this AugAssign
        # node
        # TODO: Figure out why
        value = PyBasicConversions().visit(value)
        if type(node.op) is ast.Add:
            return AddAssign(self.visit(node.target), value)
        if type(node.op) is ast.Sub:
            return SubAssign(self.visit(node.target), value)

    def visit_Assign(self, node):
        target = PyBasicConversions().visit(self.visit(node.targets[0]))
        value = PyBasicConversions().visit(self.visit(node.value))
        return Assign(target, value)

    def visit_Name(self, node):
        if node.id in self.constants.keys():
            return Constant(self.constants[node.id])
        raise Exception("Undeclared name %s. \
                Please add it to your kernel's self.constants" % node.id)
        return node
