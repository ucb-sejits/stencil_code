from ctree.omp.nodes import *
from ctree.omp.macros import *
from ctree.cpp.nodes import CppDefine
from .stencil_backend import *
from ctypes import c_int, c_double
from ctree.util import strides


class StencilOmpTransformer(StencilBackend):  #pragma: no cover
    # noinspection PyPep8Naming
    def visit_CFile(self, node):
        node.config_target = 'omp'
        # Assumes only one node in body, TODO: Can this be done?
        node.body = self.visit(node.body[0])
        return node

    # noinspection PyPep8Naming
    def visit_FunctionDecl(self, node):
        self.function_decl_helper(node)

        for index, arg in enumerate(self.input_grids + (self.input_grids[0],)):
            def_name = "_%s_array_macro" % node.params[index].name
            calc = "((_d%d)" % (arg.ndim - 1)
            for x in range(arg.ndim - 1):
                ndim = str(int(strides(arg)[x]/arg.itemsize))
                calc += "+((_d%s) * %s)" % (str(x), ndim)
            calc += ")"
            params = ["_d"+str(x) for x in range(arg.ndim)]
            node.defn.insert(0, CppDefine(def_name, params, calc))
        for index, arg in enumerate(self.arg_cfg + (self.arg_cfg[0],)):
            node.params[index].type = np.ctypeslib.ndpointer(arg.dtype, arg.ndim, arg.shape)()

        abs_decl = FunctionDecl(
            c_int(), SymbolRef('abs'), [SymbolRef('n', c_int())]
        )
        macro = CppDefine("min", [SymbolRef('_a'), SymbolRef('_b')],
                          TernaryOp(Lt(SymbolRef('_a'), SymbolRef('_b')),
                          SymbolRef('_a'), SymbolRef('_b')))
        node.params.append(SymbolRef('duration', ct.POINTER(ct.c_float)()))
        start_time = Assign(SymbolRef('start_time', c_double()), omp_get_wtime())
        node.defn.insert(0, start_time)
        end_time = Assign(Deref(SymbolRef('duration')),
                          Sub(omp_get_wtime(), SymbolRef('start_time')))
        node.defn.append(end_time)
        return [IncludeOmpHeader(), abs_decl, macro, node]

    # noinspection PyPep8Naming
    def visit_InteriorPointsLoop(self, node):
        output_grid = self.input_grids[0]
        dim = len(output_grid.shape)
        self.kernel_target = node.target
        curr_node = None
        ret_node = None
        for d in range(dim):
            target = SymbolRef(self.gen_fresh_var())
            self.var_list.append(target.name)
            for_loop = For(
                Assign(SymbolRef(target.name, c_int()),
                       Constant(self.ghost_depth[d])),
                LtE(target,
                    Constant(
                        output_grid.shape[d] -
                        self.ghost_depth[d] - 1)
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
    # noinspection PyPep8Naming
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
            else:
                pt = list(map(lambda x, y: Add(SymbolRef(x), Constant(y)),
                              self.var_list, self.offset_list))
                index = self.gen_array_macro(grid_name, pt)
                return ArrayRef(SymbolRef(grid_name), index)
        elif isinstance(target, FunctionCall) or \
                isinstance(target, MathFunction):
            return ArrayRef(SymbolRef(grid_name), self.visit(target))
        raise StencilException(
            "Unsupported GridElement encountered: {} type {} {}".format(grid_name, type(target), repr(target)))
