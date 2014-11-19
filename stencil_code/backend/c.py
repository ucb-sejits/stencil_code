__author__ = 'leonardtruong'
from ctree.cpp.nodes import CppDefine
from ctree.templates.nodes import StringTemplate
from .stencil_backend import *
from ctypes import c_int, POINTER, c_float


class StencilCTransformer(StencilBackend):
    def visit_FunctionDecl(self, node):
        super(StencilCTransformer, self).visit_FunctionDecl(node)
        for index, arg in enumerate(self.input_grids + (self.output_grid,)):
            defname = "_%s_array_macro" % node.params[index].name
            params = ','.join(["_d"+str(x) for x in range(arg.ndim)])
            params = "(%s)" % params
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
        max_macro = CppDefine("max", [SymbolRef('_a'), SymbolRef('_b')],
                          TernaryOp(Gt(SymbolRef('_a'), SymbolRef('_b')),
                          SymbolRef('_b'), SymbolRef('_a')))
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
        if this kernel is clamped then, iterate over all points and use clamping in the
        input array references on the kernel code
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
            if self.is_clamped:
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
                        pt = list(map(lambda d: self.gen_clamped_index(self.var_list[d], grid.shape[d]-1), range(len(self.var_list))))
                    else:
                        pt = list(map(lambda x: SymbolRef(x), self.var_list))

                    index = self.gen_array_macro(grid_name, pt)
                    return ArrayRef(SymbolRef(grid_name), index)
            # elif grid_name == self.neighbor_grid_name:
            else:
                if self.is_clamped:
                    grid = self.input_dict[grid_name]
                    pt = list(map(
                        lambda d, y: self.gen_clamped_index(
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
        raise Exception("Found GridElement that is not supported")

    def gen_clamped_index(self, symbol_ref, max_index):
        return FunctionCall('clamp', [symbol_ref, Constant(0), Constant(max_index)])


