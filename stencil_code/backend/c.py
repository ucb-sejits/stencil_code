__author__ = 'leonardtruong'

from ctree.cpp.nodes import CppDefine
from stencil_code.backend.stencil_backend import *
from ctree.util import strides


class StencilCTransformer(StencilBackend):
    # noinspection PyPep8Naming
    def visit_FunctionDecl(self, node):
        self.function_decl_helper(node)

        for index, arg in enumerate(self.input_grids + (self.input_grids[0],)):
        #for index, arg in enumerate(self.input_grids):
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
