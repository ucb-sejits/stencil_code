from copy import deepcopy

from ctree.c.types import *
from ctree.c.nodes import *
from ctree.ocl.nodes import *
from ctree.ocl.macros import *
from ctree.cpp.nodes import *
from ctree.visitors import NodeTransformer
from stencil_model import *
from stencil_grid import *
from ctree.templates.nodes import StringTemplate


class StencilOclTransformer(NodeTransformer):
    def __init__(self, input_grids=None, output_grid=None, kernel=None):
        if input_grids is not None:
            self.input_grids = input_grids
            self.output_grid = output_grid
            self.ghost_depth = output_grid.ghost_depth
            self.constants = kernel.constants
            self.distance = kernel.distance
        else:  # pragma: no cover
            self._instantiate_defaults()
        self.next_fresh_var = 0
        self.output_index = None
        self.neighbor_grid_name = None
        self.kernel_target = None
        self.offset_list = None
        self.var_list = []
        self.input_dict = {}
        self.input_names = []
        super(StencilOclTransformer, self).__init__()

    def _instantiate_defaults(self):  # pragma: no cover
        width = 16
        self.input_grids = [StencilGrid([width, width])]
        self.output_grid = StencilGrid([width, width])
        self.output_grid.ghost_depth = 1
        self.constants = {}


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
        # node.params.append(SymbolRef('block', _local=True))
        node.name = 'stencil_kernel'
        # for param in node.params[0:-2]:
        for param in node.params[0:-1]:
            param.set_global()
            param.set_const()
        node.params[-1].set_global()
        node.params.append(SymbolRef('block', node.params[0].type))
        node.params[-1].set_local()
        return node

    # def gen_fresh_var(self):
    #     self.next_fresh_var += 1
    #     return "x%d" % self.next_fresh_var

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
        for d in reversed(range(dim - 1)):
            index = "(" + index + ") * %d" % self.output_grid.shape[d]
            index += " + d%d" % d
        return index

    def local_array_macro(self, point):
        dim = len(self.output_grid.shape)
        index = get_local_id(dim)
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
        index = "d%d" % (dim - 1)
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

    # def gen_block_index(self):
    #     dim = len(self.output_grid.shape)
    #     index = Add(
    #         get_local_id(dim - 1),
    #         Constant(self.ghost_depth)
    #     )
    #     for d in reversed(range(dim - 1)):
    #         index = Add(
    #             Mul(
    #                 index,
    #                 Add(
    #                     get_local_size(d),
    #                     Constant(2 * self.ghost_depth)
    #                 ),
    #             ),
    #             Add(get_local_id(d), self.ghost_depth)
    #         )
    #     return index

    def visit_InteriorPointsLoop(self, node):
        dim = len(self.output_grid.shape)
        self.kernel_target = node.target
        body = []

        global_index = [SymbolRef("id%d" % d) for d in range(dim)]
        global_idx = SymbolRef('global_index')
        self.output_index = global_idx
        for d in range(dim):
            body.append(
                Assign(
                    SymbolRef('id%d' % d, UInt()),
                    Add(get_global_id(d), self.ghost_depth)
                )
            )
        body.append(Assign(SymbolRef('global_index', UInt()),
                    self.gen_global_index()))

        local_index = [Add(get_local_id(index), Constant(self.ghost_depth)) for
                       index in range(dim)]
        local_idx = self.local_array_macro(local_index)

        # body = []
        # for d in range(dim):
        #     body.append(
        #         Assign(SymbolRef('id%d' % d, UInt()), get_global_id(d))
        #     )
        # body.append(assign(symbolref('global_index', uint()),
        #             self.gen_global_index()))
        # body.append(Assign(SymbolRef('blk_index', UInt()),
        #             self.gen_block_index()))
        # body.append(
        #     Assign(
        #         ArrayRef(
        #             SymbolRef('block'),
        #             SymbolRef('blk_index')
        #         ),
        #         ArrayRef(
        #             SymbolRef(self.input_names[0]),
        #             SymbolRef('global_index'))
        #         )
        # )
        body.append(
            CppDefine("local_array_macro",["d%d" % i for i in range(dim)],
            self.gen_local_macro())
        )
        body.append(
            CppDefine("global_array_macro", ["d%d" % i for i in range(dim)],
            self.gen_global_macro())
        )
        local_size_total = get_local_size(0)
        curr_node = For(Assign(SymbolRef('d%d' % (dim - 1), Int()),
                               get_local_id(dim - 1)),
                        Lt(
                            SymbolRef('d%d' % (dim - 1)),
                            Add(get_local_size(dim - 1),
                                Constant(self.ghost_depth * 2)
                            )
                        ),
                        AddAssign(
                            SymbolRef('d%d' % (dim - 1)), get_local_size(dim - 1)
                        ),
                        []
            )
        body.append(curr_node)
        for d in reversed(range(0, dim - 1)):
            curr_node.body.append(
                For(
                    Assign(SymbolRef('d%d' % d, Int()), get_local_id(d)),
                    Lt(
                        SymbolRef('d%d' % d),
                        Add(
                            get_local_size(d), Constant(self.ghost_depth * 2)
                        )
                    ),
                    AddAssign(SymbolRef('d%d' %d), get_local_size(d)),
                    []
                )
            )
            curr_node = curr_node.body[0]

        curr_node.body.append(Assign(
                 ArrayRef(
                    SymbolRef('block'),
                    self.local_array_macro(
                        [SymbolRef("d%d" % d) for d in range(0, dim)]
                    )
                ),
                ArrayRef(
                    SymbolRef(self.input_names[0]),
                    self.global_array_macro(
                        [Add(
                            SymbolRef("d%d" % d),
                            Mul(get_group_id(d), get_local_size(d))
                        ) for d in range(0, dim)]
                    )
                )
            )
        )

        body.append(FunctionCall(SymbolRef("barrier"), [SymbolRef("CLK_LOCAL_MEM_FENCE")]))
        for d in range(0, dim):
            body.append(Assign(SymbolRef('local_id%d' % d, UInt()), get_local_id(d)))
            self.var_list.append("local_id%d" % d)

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
                pt = list(map(lambda x, y: Add(Add(SymbolRef(x), SymbolRef(y)), Constant(self.ghost_depth)),
                              self.var_list, self.offset_list))
                #index = self.gen_array_macro(grid_name, pt)
                index = self.local_array_macro(pt)
                #index = SymbolRef('out_index')
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
