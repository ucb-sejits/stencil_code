from copy import deepcopy

from ctree.c.nodes import *
from ctree.ocl.macros import *
from ctree.cpp.nodes import *
from ctree.templates.nodes import StringTemplate
from ..stencil_model import *
from stencil_backend import StencilBackend
import numpy as np
import ctypes as ct


class StencilOclSemanticTransformer(StencilBackend):
    def __init__(self, input_grids=None, output_grid=None, kernel=None,
            fusion_padding=None):
        super(StencilOclSemanticTransformer, self).__init__(input_grids, output_grid,
                kernel)
        self.fusion_padding = fusion_padding

    def visit_FunctionDecl(self, node):
        # This function grabs the input and output grid names which are used to
        # generate the proper array macros.
        super(StencilOclSemanticTransformer, self).visit_FunctionDecl(node)
        for index, param in enumerate(node.params[:-1]):
            # TODO: Transform numpy type to ctype
            arg = self.input_grids[index]
            param.type = ct.POINTER(ct.c_float)()
            param.set_global()
            param.set_const()
        node.set_kernel()
        node.params[-1].set_global()
        arg = self.output_grid
        node.params[-1].type = ct.POINTER(ct.c_float)()
        # node.params.append(SymbolRef('block', np.ctypeslib.ndpointer(arg.data.dtype, arg.data.ndim, arg.data.shape)()))
        node.params.append(SymbolRef('block', ct.POINTER(ct.c_float)()))
        node.params[-1].set_local()
        node.defn = node.defn[0]
        return node

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

    def gen_local_macro(self):
        dim = len(self.output_grid.shape)
        index = "d%d" % (dim - 1)
        for d in reversed(range(dim - 1)):
            index = "(" + index + ") * (get_local_size(%d) + %d)" % (d, 2 *
                    self.ghost_depth * self.fusion_padding)
            index += " + d%d" % d
        return index

    def load_shared_memory_block(self, target, padding):
        dim = len(self.output_grid.shape)
        decls = []
        thread_id = get_local_id(0)
        num_threads = get_local_size(0)
        block_size = Add(get_local_size(0), padding)
        for d in range(1, dim):
            thread_id = Add(
                    Mul(get_local_id(d), get_local_size(d - 1)),
                    thread_id
                )
            num_threads = Mul(get_local_size(d), num_threads)
            block_size = Mul(
                Add(get_local_size(d), padding),
                block_size
            )

        decls.append(Assign(SymbolRef("thread_id", ct.c_int()), thread_id))
        decls.append(Assign(SymbolRef("block_size", ct.c_int()), block_size))
        decls.append(Assign(SymbolRef("num_threads", ct.c_int()), num_threads))
        base = None
        for i in reversed(range(0, dim - 1)):
            if base is not None:
                base = Mul(Add(get_local_size(i), padding), base)
            else:
                base = Add(get_local_size(i), padding)
        local_indices = [
                    Assign(
                        SymbolRef("local_id%d" % (dim - 1), ct.c_int()),
                        Div(SymbolRef('tid'), base)
                        ),
                    Assign(
                        SymbolRef("r_%d" % (dim - 1), ct.c_int()),
                        Mod(SymbolRef('tid'), base)
                        )
                ]
        for d in reversed(range(0, dim - 1)):
            base = None
            for i in reversed(range(0, d - 1)):
                if base is not None:
                    base = Mul(Add(get_local_size(i), padding), base)
                else:
                    base = Add(get_local_size(i), padding)
            if base is not None:
                local_indices.append(
                    Assign(
                        SymbolRef("local_id%d" % d, ct.c_int()),
                        Div(SymbolRef('r_%d' % (d + 1)), base)
                        )
                    )
                local_indices.append(
                    Assign(
                        SymbolRef("r_%d" % d, ct.c_int()),
                        Mod(SymbolRef('r_%d' % (d + 1)), base)
                        )
                    )
            else:
                local_indices.append(
                    Assign(
                        SymbolRef("local_id%d" % d, ct.c_int()),
                        SymbolRef('r_%d' % (d + 1))
                        )
                    )
        body = For(
                Assign(SymbolRef('tid', ct.c_int()), SymbolRef('thread_id')),
                Lt(SymbolRef('tid'), SymbolRef('block_size')),
                AddAssign(SymbolRef('tid'), SymbolRef('num_threads')),
                local_indices + [Assign(
                 ArrayRef(
                    target,
                    SymbolRef('tid')
                ),
                ArrayRef(
                    SymbolRef(self.input_names[0]),
                    self.global_array_macro(
                        [Add(
                            SymbolRef("local_id%d" % d),
                            Mul(FunctionCall(SymbolRef('get_group_id'),[Constant(d)]), get_local_size(d))
                        ) for d in range(0, dim)]
                    )
                )
                )]
            )
        return decls, [body, barrier(CLK_LOCAL_MEM_FENCE())]

    def visit_InteriorPointsLoop(self, node):
        dim = len(self.output_grid.shape)
        self.kernel_target = node.target
        body = []

        body.append(MacroDefns([
            CppDefine("local_array_macro",["d%d" % i for i in range(dim)],
            self.gen_local_macro()),
            CppDefine("global_array_macro", ["d%d" % i for i in range(dim)],
            self.gen_global_macro())]
        ))
        body.append(LoadSharedMemBlock(*self.load_shared_memory_block(SymbolRef('block'),
            Constant(self.ghost_depth * 2 * self.fusion_padding))))

        self.output_index = SymbolRef('global_index')
        next_body = []
        for d in range(0, dim):
            next_body.append(Assign(SymbolRef('local_id%d' % d, ct.c_int()),
                               Add(get_local_id(d), Constant(self.ghost_depth))))
            self.var_list.append("local_id%d" % d)
        map(next_body.extend, map(self.visit, node.body))
        body.append(OclNeighborLoop(next_body, self.output_grid.shape,
            self.ghost_depth))
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
                    # index = self.gen_array_macro(grid_name, pt)
                    index = self.local_array_macro(pt)
                    return ArrayRef(SymbolRef('block'), index)
            elif grid_name == self.neighbor_grid_name:
                pt = list(map(lambda x, y: Add(SymbolRef(x), SymbolRef(y)),
                              self.var_list, self.offset_list))
                #index = self.gen_array_macro(grid_name, pt)
                index = self.local_array_macro(pt)
                #index = SymbolRef('out_index')
                return ArrayRef(SymbolRef('block'), index)
        elif isinstance(target, FunctionCall) or \
                isinstance(target, MathFunction):
            return ArrayRef(SymbolRef(grid_name), self.visit(target))
        return node

class StencilOclTransformer(StencilBackend):
    def __init__(self, input_grids=None, output_grid=None, kernel=None,
            block_padding=None):
        super(StencilOclTransformer, self).__init__(input_grids, output_grid,
                kernel)
        self.block_padding = block_padding

    def visit_FunctionDecl(self, node):
        # This function grabs the input and output grid names which are used to
        # generate the proper array macros.
        super(StencilOclTransformer, self).visit_FunctionDecl(node)
        for index, param in enumerate(node.params[:-1]):
            # TODO: Transform numpy type to ctype
            arg = self.input_grids[index]
            param.type = ct.POINTER(ct.c_float)()
            param.set_global()
            param.set_const()
        node.set_kernel()
        node.params[-1].set_global()
        arg = self.output_grid
        node.params[-1].type = ct.POINTER(ct.c_float)()
        # node.params.append(SymbolRef('block', np.ctypeslib.ndpointer(arg.data.dtype, arg.data.ndim, arg.data.shape)()))
        node.params.append(SymbolRef('block', ct.POINTER(ct.c_float)()))
        node.params[-1].set_local()
        node.defn = node.defn[0]
        return node

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


    def gen_array_macro(self, arg, point):
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
        index = Add(get_global_id(dim - 1), Constant(self.ghost_depth))
        for d in reversed(range(dim - 1)):
            index = Add(
                Mul(
                    index,
                    Constant(self.output_grid.shape[d])
                ),
                Add(get_global_id(d), Constant(self.ghost_depth))
            )
        return index

    def load_shared_memory_block(self, target, padding):
        dim = len(self.output_grid.shape)
        body = []
        thread_id = get_local_id(0)
        num_threads = get_local_size(0)
        block_size = Add(get_local_size(0), padding)
        for d in range(1, dim):
            thread_id = Add(
                    Mul(get_local_id(d), get_local_size(d - 1)),
                    thread_id
                )
            num_threads = Mul(get_local_size(d), num_threads)
            block_size = Mul(
                Add(get_local_size(d), padding),
                block_size
            )

        body.append(Assign(SymbolRef("thread_id", ct.c_int()), thread_id))
        body.append(Assign(SymbolRef("block_size", ct.c_int()), block_size))
        body.append(Assign(SymbolRef("num_threads", ct.c_int()), num_threads))
        base = None
        for i in reversed(range(0, dim - 1)):
            if base is not None:
                base = Mul(Add(get_local_size(i), Constant(self.ghost_depth *
                    2)), base)
            else:
                base = Add(get_local_size(i), Constant(self.ghost_depth *
                    2))
        if base is not None:
            local_indices = [
                        Assign(
                            SymbolRef("local_id%d" % (dim - 1), ct.c_int()),
                            Div(SymbolRef('tid'), base)
                            ),
                        Assign(
                            SymbolRef("r_%d" % (dim - 1), ct.c_int()),
                            Mod(SymbolRef('tid'), base)
                            )
                    ]
        else:
            local_indices = [
                Assign(
                    SymbolRef("local_id%d" % (dim - 1), ct.c_int()),
                    SymbolRef('tid')
                ),
                Assign(
                    SymbolRef("r_%d" % (dim - 1), ct.c_int()),
                    SymbolRef('tid')
                )
            ]
        base = None
        for d in reversed(range(0, dim - 1)):
            for i in reversed(range(0, d)):
                if base is not None:
                    base = Mul(Add(get_local_size(i), padding), base)
                else:
                    base = Add(get_local_size(i), padding)
            if base is not None:
                local_indices.append(
                    Assign(
                        SymbolRef("local_id%d" % d, ct.c_int()),
                        Div(SymbolRef('r_%d' % (d + 1)), base)
                        )
                    )
                local_indices.append(
                    Assign(
                        SymbolRef("r_%d" % d, ct.c_int()),
                        Mod(SymbolRef('r_%d' % (d + 1)), base)
                        )
                    )
            else:
                local_indices.append(
                    Assign(
                        SymbolRef("local_id%d" % d, ct.c_int()),
                        SymbolRef('r_%d' % (d + 1))
                        )
                    )
        body.append(
            For(
                Assign(SymbolRef('tid', ct.c_int()), SymbolRef('thread_id')),
                Lt(SymbolRef('tid'), SymbolRef('block_size')),
                AddAssign(SymbolRef('tid'), SymbolRef('num_threads')),
                local_indices + [Assign(
                 ArrayRef(
                    target,
                    SymbolRef('tid')
                ),
                ArrayRef(
                    SymbolRef(self.input_names[0]),
                    self.global_array_macro(
                        [Add(
                            SymbolRef("local_id%d" % d),
                            Mul(FunctionCall(SymbolRef('get_group_id'),[Constant(d)]), get_local_size(d))
                        ) for d in range(0, dim)]
                    )
                )
                )]
            )
        )
        return body

    def visit_InteriorPointsLoop(self, node):
        dim = len(self.output_grid.shape)
        self.kernel_target = node.target
        body = []

        body.append(
            CppDefine("local_array_macro",["d%d" % i for i in range(dim)],
            self.gen_local_macro())
        )
        body.append(
            CppDefine("global_array_macro", ["d%d" % i for i in range(dim)],
            self.gen_global_macro())
        )

        global_idx = SymbolRef('global_index')
        self.output_index = global_idx
        body.append(Assign(SymbolRef('global_index', ct.c_int()),
                    self.gen_global_index()))

        body.extend(self.load_shared_memory_block(SymbolRef('block'),
            Constant(self.ghost_depth * 2)))
        body.append(FunctionCall(SymbolRef("barrier"), [SymbolRef("CLK_LOCAL_MEM_FENCE")]))
        for d in range(0, dim):
            body.append(Assign(SymbolRef('local_id%d' % d, ct.c_int()),
                               Add(get_local_id(d), Constant(self.ghost_depth))))
            self.var_list.append("local_id%d" % d)

        for child in map(self.visit, node.body):
            if isinstance(child, list):
                body.extend(child)
            else:
                body.append(child)
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
                    # index = self.gen_array_macro(grid_name, pt)
                    index = self.local_array_macro(pt)
                    return ArrayRef(SymbolRef('block'), index)
            else:
                pt = list(map(lambda x, y: Add(SymbolRef(x), SymbolRef(y)),
                              self.var_list, self.offset_list))
                #index = self.gen_array_macro(grid_name, pt)
                index = self.local_array_macro(pt)
                #index = SymbolRef('out_index')
                return ArrayRef(SymbolRef('block'), index)
        elif isinstance(target, FunctionCall) or \
                isinstance(target, MathFunction):
            return ArrayRef(SymbolRef(grid_name), self.visit(target))
        print(self.input_dict)
        raise Exception(
            "Unsupported GridElement encountered: {0}".format(grid_name))
