__author__ = 'leonardtruong'
from ctree.cpp.nodes import CppDefine
from ctypes import POINTER, c_float
from stencil_code.stencil_exception import StencilException
from stencil_code.backend.stencil_backend import *


class StencilCTransformer(StencilBackend):
    def visit_FunctionDecl(self, node):
        super(StencilCTransformer, self).visit_FunctionDecl(node)
        for index, arg in enumerate(self.input_grids + (self.output_grid,)):
            defname = "_%s_array_macro" % node.params[index].name
            # params = ','.join(["_d"+str(x) for x in range(arg.ndim)])
            # params = "(%s)" % params
            calc = "((_d%d)" % (arg.ndim - 1)
            for x in range(arg.ndim - 1):
                ndim = str(int(arg.strides[x]/arg.itemsize))
                calc += "+((_d%s) * %s)" % (str(x), ndim)
            calc += ")"
            params = ["_d"+str(x) for x in range(arg.ndim)]
            node.defn.insert(0, CppDefine(defname, params, calc))
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
        node.params.append(SymbolRef('duration', POINTER(c_float)))
        start_time = Assign(StringTemplate('clock_t start_time'), FunctionCall(
            SymbolRef('clock')))
        node.defn.insert(0, start_time)
        end_time = Assign(Deref(SymbolRef('duration')),
                          Div(Sub(FunctionCall(SymbolRef('clock')), SymbolRef(
                              'start_time')), SymbolRef('CLOCKS_PER_SEC')))
        node.defn.append(end_time)
        return [StringTemplate("#include <time.h>"), abs_decl, min_macro, clamp_macro, node]

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
        dim = len(self.output_grid.shape)
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
                            self.output_grid.shape[d] - 1)
                        ),
                    PostInc(SymbolRef(target)),
                    [])
            else:
                for_loop = For(
                    Assign(SymbolRef(target, c_int()),
                           Constant(self.ghost_depth[d])),
                    LtE(SymbolRef(target),
                        Constant(
                            self.output_grid.shape[d] -
                            self.ghost_depth[d] - 1)
                        ),
                    PostInc(SymbolRef(target)),
                    [])

            if d == 0:
                ret_node = for_loop
            else:
                curr_node.body = [for_loop]
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
                       Constant(self.output_grid.shape[index] - (self.ghost_depth[index] + 1))),
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
        target = node.target
        if isinstance(target, SymbolRef):
            target = target.name
            if target == self.kernel_target:
                if grid_name is self.output_grid_name:
                    return ArrayRef(SymbolRef(self.output_grid_name),
                                    SymbolRef(self.output_index))
                elif grid_name in self.input_dict:
                    # grid = self.input_dict[grid_name]
                    if self.is_clamped:
                        grid = self.input_dict[grid_name]
                        pt = list(
                            map(lambda d: gen_clamped_index(
                                self.var_list[d], grid.shape[d]-1), range(len(self.var_list))))
                    else:  # pragma no cover
                        pt = list(map(lambda x: SymbolRef(x), self.var_list))

                    index = self.gen_array_macro(grid_name, pt)
                    return ArrayRef(SymbolRef(grid_name), index)
            # elif grid_name == self.neighbor_grid_name:
            else:
                if self.is_clamped:
                    grid = self.input_dict[grid_name]
                    pt = list(map(
                        lambda d, y: gen_clamped_index(
                            Add(SymbolRef(self.var_list[d]), Constant(y)),
                            grid.shape[d]-1), range(len(self.var_list)), self.offset_list
                    ))
                else:
                    pt = list(map(lambda x, y: Add(SymbolRef(x), Constant(y)),
                              self.var_list, self.offset_list))
                index = self.gen_array_macro(grid_name, pt)
                return ArrayRef(SymbolRef(grid_name), index)
        elif isinstance(target, FunctionCall) or \
                isinstance(target, MathFunction):
            return ArrayRef(SymbolRef(grid_name), self.visit(target))
        raise StencilException("Found GridElement that is not supported")  # pragma no cover
