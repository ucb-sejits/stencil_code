__author__ = 'leonardtruong'
from ctree.cpp.nodes import CppDefine
from ctree.templates.nodes import StringTemplate
from stencil_backend import *
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
        macro = CppDefine("min", [SymbolRef('_a'), SymbolRef('_b')],
                          TernaryOp(Lt(SymbolRef('_a'), SymbolRef('_b')),
                          SymbolRef('_a'), SymbolRef('_b')))
        node.params.append(SymbolRef('duration', POINTER(c_float)))
        start_time = Assign(StringTemplate('clock_t start_time'), FunctionCall(
            SymbolRef('clock')))
        node.defn.insert(0, start_time)
        end_time = Assign(Deref(SymbolRef('duration')),
                          Div(Sub(FunctionCall(SymbolRef('clock')), SymbolRef(
                              'start_time')), SymbolRef('CLOCKS_PER_SEC')))
        node.defn.append(end_time)
        return [StringTemplate("#include <time.h>"), abs_decl, macro, node]

    def visit_InteriorPointsLoop(self, node):
        dim = len(self.output_grid.shape)
        self.kernel_target = node.target
        curr_node = None
        ret_node = None
        for d in range(dim):
            target = self.gen_fresh_var()
            self.var_list.append(target)
            for_loop = For(
                Assign(SymbolRef(target, c_int()),
                       Constant(self.ghost_depth)),
                LtE(SymbolRef(target),
                    Constant(
                        self.output_grid.shape[d] -
                        self.ghost_depth - 1)
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
            # elif grid_name == self.neighbor_grid_name:
            else:
                pt = list(map(lambda x, y: Add(SymbolRef(x), Constant(y)),
                              self.var_list, self.offset_list))
                index = self.gen_array_macro(grid_name, pt)
                return ArrayRef(SymbolRef(grid_name), index)
        elif isinstance(target, FunctionCall) or \
                isinstance(target, MathFunction):
            return ArrayRef(SymbolRef(grid_name), self.visit(target))
        raise Exception("Found GridElement that is not supported")

