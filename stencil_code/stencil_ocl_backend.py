from copy import deepcopy

from ctree.c.types import *
from ctree.c.nodes import *
from ctree.ocl.nodes import *
from ctree.ocl.macros import *
from ctree.visitors import NodeTransformer
from stencil_model import *


### TODO: ADD THESE TO CTREE ###
def get_global_id(idx):
    return FunctionCall(SymbolRef("get_global_id"), [Constant(idx)])

def get_local_id(idx):
    return FunctionCall(SymbolRef("get_local_id"), [Constant(idx)])

def get_group_id(idx):
    return FunctionCall(SymbolRef("get_group_id"), [Constant(idx)])

def get_local_size(idx):
    return FunctionCall(SymbolRef("get_local_size"), [Constant(idx)])

def get_num_groups(idx):
    return FunctionCall(SymbolRef("get_num_groups"), [Constant(idx)])
#############

class StencilOclTransformer(NodeTransformer):
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
        self.input_names = []
        super(StencilOclTransformer, self).__init__()

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
        node.defn = self.visit(node.defn[0])
        node.kernel = True
        return node

    def gen_fresh_var(self):
        self.next_fresh_var += 1
        return "x%d" % self.next_fresh_var


    def global_array_macro(self, point):
        dim = len(self.output_grid.shape)
        index = point[dim - 1]
        for d in reversed(range(dim - 1)):
            index = Add(
                Mul(
                    index,
                    Constant(self.output_grid.shape[d])
                ),
                point[d]
            )

        return FunctionCall(SymbolRef("global_array_macro"), point)

    def gen_global_macro(self):
        dim = len(self.output_grid.shape)
        index = "d%d" % (dim - 1)
        for d in reversed(range(dim)):
            index = "(" + index + ") * %d" % self.output_grid.shape[d]
            index += " + d%d" % d
        return index

    def local_array_macro(self, point):
        dim = len(self.output_grid.shape)
        index = Add(
            get_local_id(dim),
            Constant(self.ghost_depth)
        )
        for d in reversed(range(dim)):
            index = Add(
                Mul(
                    index,
                    Add(
                        get_local_size(d),
                        Constant(2 * self.ghost_depth)
                    ),
                ),
                point[d]
            )
        return FunctionCall(SymbolRef("local_array_macro"), point)

    def gen_local_macro(self):
        dim = len(self.output_grid.shape)
        index = "d%d + %d" % (dim - 1, self.ghost_depth)
        for d in reversed(range(dim - 1)):
            index = "(" + index + ") * (get_local_size(%d) + %d)" % (d, 2 * self.ghost_depth)
            index += " + d%d" % d
        return index

    def gen_global_index(self):
        dim = len(self.output_grid.shape)
        index = SymbolRef("id%d" % (dim - 1))
        for d in reversed(range(dim - 1)):
            index = Add(
                Mul(
                    index,
                    Constant(self.output_grid.shape[d])
                ),
                SymbolRef("id%d" % d)
            )
        return index

    def gen_block_index(self):
        dim = len(self.output_grid.shape)
        index = Add(
            get_local_id(dim - 1),
            Constant(self.ghost_depth)
        )
        for d in reversed(range(dim - 1)):
            index = Add(
                Mul(
                    index,
                    Add(
                        get_local_size(d),
                        Constant(2 * self.ghost_depth)
                    ),
                ),
                Add(get_local_id(d), self.ghost_depth)
            )
        return index

    def visit_InteriorPointsLoop(self, node):
        dim = len(self.output_grid.shape)
        self.kernel_target = node.target

        global_index = [get_global_id(index) for index in range(dim)]
        global_idx = SymbolRef('out_index')
        self.output_index = global_idx

        local_index = [Add(get_local_id(index), Constant(self.ghost_depth)) for
                       index in range(dim)]
        local_idx = self.local_array_macro(local_index)

        body = []
        for d in range(dim):
            body.append(Assign(SymbolRef('id%d' % d, UInt()), get_global_id(d)))
        body.append(Assign(SymbolRef('out_index', UInt()),
                    self.gen_global_index()))
        body.append(Assign(SymbolRef('blk_index', UInt()),
                    self.gen_block_index()))
        body.append(Define("local_array_macro", ["d%d" % i for i in range(dim)],
                    self.gen_local_macro()))
        body.append(Define("global_array_macro", ["d%d" % i for i in range(dim)],
                    self.gen_global_macro()))
        for d in range(0, dim):
            self.var_list.append(get_local_id(d))
            body1 = []
            body2 = []
            for x in range(1, self.ghost_depth + 1):
                block_idx1 = self.local_array_macro(local_index[:d] +
                                               [Add(get_local_id(d),
                                                    Constant(x))] + local_index[d+1:])

                block_idx2 = self.local_array_macro(local_index[:d] +
                                               [Add(get_local_id(d),
                                                    Constant(x + self.ghost_depth + 1))] +
                                               local_index[d+1:])

                target1 = global_index[:d] + global_index[d+1:]
                target2 = global_index[:d] + global_index[d+1:]
                target1.insert(d,
                    Sub(
                        Mul(
                            get_local_size(d),
                            get_group_id(d)
                        ),
                        Constant(x)
                    )
                )
                target2.insert(d,
                    Mul(get_local_size(d), Add(get_group_id(d), Constant(x)))
                )
                input_idx1 = self.global_array_macro(target1)
                input_idx2 = self.global_array_macro(target2)
                body1.append(
                    Assign(
                        ArrayRef(SymbolRef('block'), block_idx1),
                        ArrayRef(SymbolRef(self.input_names[0]), input_idx1)
                    )
                )
                body2.append(
                    Assign(
                        ArrayRef(SymbolRef('block'), block_idx2),
                        ArrayRef(SymbolRef(self.input_names[0]), input_idx2)
                    )
                )
            body.append(If(Eq(get_local_id(d), Constant(0)), body1))
            body.append(If(Eq(get_local_id(d), Sub(get_local_size(d), Constant(1))), body2))

            # base_corner = [Sub(Add(Mul(get_local_size(i), get_group_id(i)), getlocal_size(i)),
            # Constant(1 - dim)) for i in range(dim)]
            # corner1 = self.global_array_macro(base_corner)
            # second_corner = base_corner[:d] + [Mul(get_local_size(i), get_group_id(i) for i in
            #                                        range(d,dim)]
            # corner2 = self.global_array_macro(second_corner)

        body.append(FunctionCall(SymbolRef("barrier"),[SymbolRef("CLK_LOCAL_MEM_FENCE")]))
        for child in map(self.visit, node.body):
            if isinstance(child, list):
                body.extend(child)
            else:
                body.append(child)
        return body

    def visit_NeighborPointsLoop(self, node):
        dim = len(self.output_grid.shape)
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
                #index = self.gen_array_macro(grid_name, pt)
                index = self.local_array_macro(pt)
                return ArrayRef(SymbolRef('block'), index)
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
            return Constant(self.constants[node.id])
        return node
