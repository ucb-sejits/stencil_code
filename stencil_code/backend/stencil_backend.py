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
        self.ghost_depth = parent_lazy_specializer.ghost_depth if parent_lazy_specializer else 0
        self.is_clamped = parent_lazy_specializer.is_clamped if parent_lazy_specializer else False
        self.is_copied = parent_lazy_specializer.is_copied if parent_lazy_specializer else False
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
        self.constants = parent_lazy_specializer.constants if parent_lazy_specializer else []
        self.distance = parent_lazy_specializer.distance if parent_lazy_specializer else None
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
    def visit_InteriorPointsLoop(self, node):
        """
        generate the c for loops necessary to represent the interior points iteration
        for boundary_handling
        if clamped then, iterate over all points and use clamping in the
            input array references on the kernel code
        if copied then
            insert an if before the unrolled neighbor stuff to do a direct copy from
            the original input_grid
        :param node:
        :return:
        """
        output_grid_shape = self.input_grids[0].shape
        dim = len(output_grid_shape)
        self.kernel_target = node.target
        curr_node = None
        ret_node = None
        for d in range(dim):
            target = self.gen_fresh_var()
            self.var_list.append(target)
            if self.is_clamped or self.is_copied:
                for_loop = For(
                    Assign(SymbolRef(target, c_int()),
                           Constant(0)),
                    LtE(SymbolRef(target),
                        Constant(
                            output_grid_shape[d] - 1)
                        ),
                    PostInc(SymbolRef(target)),
                    [])
            else:
                for_loop = For(
                    Assign(SymbolRef(target, c_int()),
                           Constant(self.ghost_depth[d])),
                    LtE(SymbolRef(target),
                        Constant(
                            output_grid_shape[d] -
                            self.ghost_depth[d] - 1)
                        ),
                    PostInc(SymbolRef(target)),
                    [])

            if d == 0:
                ret_node = for_loop
                self.index_target_dict[node.target] = (node.grid_name, target)
            else:
                curr_node.body = [for_loop]
                self.index_target_dict[node.target] += (target,)

            curr_node = for_loop
        self.output_index = self.gen_fresh_var()
        pt = [SymbolRef(x) for x in self.var_list]
        macro = self.gen_array_macro(self.output_grid_name, pt)
        curr_node.body = [Assign(SymbolRef(self.output_index, c_int()),
                                 macro)]

        if self.is_copied:
            # this a little hokey but if we are in boundary copy mode
            # we change the for loops above to visit everything and
            # then we test to see if the any of the values are in the halo zone
            # and if so we just copy straight from in_grid, otherwise we
            # do the neighborhood unrolling
            def test_index_in_halo(index):
                return Or(
                    Lt(SymbolRef(self.var_list[index]), Constant(self.ghost_depth[index])),
                    Gt(SymbolRef(self.var_list[index]),
                       Constant(output_grid_shape[index] - (self.ghost_depth[index] + 1))),
                )

            def boundary_or(index):
                if index < len(self.var_list) - 1:
                    return Or(test_index_in_halo(index), boundary_or(index+1))
                else:
                    return test_index_in_halo(index)

            then_block = Assign(
                ArrayRef(SymbolRef(self.output_grid_name), SymbolRef(self.output_index)),
                ArrayRef(SymbolRef(self.input_names[0]), SymbolRef(self.output_index)),
            )
            else_block = []
            for elem in map(self.visit, node.body):
                if type(elem) == list:
                    else_block.extend(elem)
                else:  # pragma no cover
                    else_block.append(elem)
            if_boundary_block = If(
                boundary_or(0),
                then_block,
                else_block
            )
            curr_node.body.append(if_boundary_block)
        else:
            for elem in map(self.visit, node.body):
                if type(elem) == list:
                    curr_node.body.extend(elem)
                else:
                    curr_node.body.append(elem)

        self.kernel_target = None
        return ret_node

    # noinspection PyPep8Naming
    def visit_GridElement(self, node):
        """
        handles array references to input_grids, understands clamping
        on input_grids by looking at the kernel of the current specializer
        :param node:
        :return:
        """

        def gen_clamped_index(symbol_ref, max_index):
            return FunctionCall('clamp', [symbol_ref, Constant(0), Constant(max_index)])

        grid_name = node.grid_name
        grid = self.input_dict[grid_name]
        target = node.target

        if isinstance(target, SymbolRef):
            if target.name in self.index_target_dict:
                dict_tuple = self.index_target_dict[target.name]
                index_components = [SymbolRef(val) for val in dict_tuple[1:]]
                if target.name in self.offset_dict:
                    offsets = self.offset_dict[target.name]
                    index_components = [
                        Add(symbol, Constant(offsets[index]))
                        for index, symbol in enumerate(index_components)
                    ]
                if self.is_clamped:
                    index_components = [
                        gen_clamped_index(element, grid.shape[index]-1)
                        for index, element in enumerate(index_components)
                    ]
                return ArrayRef(SymbolRef(grid_name), self.gen_array_macro(grid_name, index_components))
            else:
                return ArrayRef(SymbolRef(grid_name), target)
        elif isinstance(target, FunctionCall) or \
                isinstance(target, MathFunction):
            return ArrayRef(SymbolRef(grid_name), self.visit(target))
        raise StencilException(
            "Unsupported GridElement encountered: {} type {} {}".format(
                grid_name, type(target), repr(target)))  # pragma no cover

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
