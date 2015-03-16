__author__ = 'leonardtruong'

from ctree.cpp.nodes import CppDefine
from stencil_code.backend.stencil_backend import *
from ctree.util import strides


class StencilCTransformer(StencilBackend):
    # noinspection PyPep8Naming
    def visit_FunctionDecl(self, node):
        self.function_decl_helper(node)

        for index, arg in enumerate(self.input_grids + (self.input_grids[0],)):
            def_name = "_%s_array_macro" % node.params[index].name
            calc = "((_d%d)" % (arg.ndim - 1)
            for x in range(arg.ndim - 1):
                ndim = str(int(strides(arg.shape)[x]))
                calc += "+((_d%s) * %s)" % (str(x), ndim)
            calc += ")"
            params = ["_d"+str(x) for x in range(arg.ndim)]
            node.defn.insert(0, CppDefine(def_name, params, calc))

        abs_decl = FunctionDecl(
            c_int(), SymbolRef('abs'), [SymbolRef('n', c_int())]
        )
        min_macro = CppDefine("min", [SymbolRef('_a'), SymbolRef('_b')],
                              TernaryOp(Lt(SymbolRef('_a'), SymbolRef('_b')),
                                        SymbolRef('_a'), SymbolRef('_b')))
        clamp_macro = CppDefine(
            "clamp", [SymbolRef('_a'), SymbolRef('_min_a'), SymbolRef('_max_a')],
            StringTemplate("(_a>_max_a?_max_a:_a)<_min_a?_min_a:(_a>_max_a?_max_a:_a)"),
        )
        node.params.append(SymbolRef('duration', ct.POINTER(ct.c_float)()))
        start_time = Assign(StringTemplate('clock_t start_time'), FunctionCall(
            SymbolRef('clock')))
        node.defn.insert(0, start_time)
        end_time = Assign(Deref(SymbolRef('duration')),
                          Div(Sub(FunctionCall(SymbolRef('clock')), SymbolRef(
                              'start_time')), SymbolRef('CLOCKS_PER_SEC')))
        node.defn.append(end_time)
        return [StringTemplate("#include <time.h>"), abs_decl, min_macro, clamp_macro, node]

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
        output_grid = self.input_grids[0]
        dim = len(output_grid.shape)
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
                            output_grid.shape[d] - 1)
                        ),
                    PostInc(SymbolRef(target)),
                    [])
            else:
                for_loop = For(
                    Assign(SymbolRef(target, c_int()),
                           Constant(self.ghost_depth[d])),
                    LtE(SymbolRef(target),
                        Constant(
                            output_grid.shape[d] -
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
                       Constant(output_grid.shape[index] - (self.ghost_depth[index] + 1))),
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
