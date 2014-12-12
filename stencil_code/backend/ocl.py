import ctypes as ct

from ctree.c.nodes import Lt, Constant, And, SymbolRef, Assign, Add, Mul, \
    Div, Mod, For, AddAssign, ArrayRef, FunctionCall, ArrayDef, Ref, \
    FunctionDecl, GtE, Sub, Cast
from ctree.ocl.macros import get_global_id, get_local_id, get_local_size, \
    clSetKernelArg, NULL
from ctree.cpp.nodes import CppDefine
from ctree.ocl.nodes import OclFile
from ctree.templates.nodes import StringTemplate
import pycl as cl

from stencil_code.stencil_exception import StencilException
from stencil_code.stencil_model import MathFunction
from stencil_code.backend.stencil_backend import StencilBackend
from stencil_code.backend.ocl_boundary_copier import boundary_kernel_factory


class StencilOclTransformer(StencilBackend):
    def __init__(self, input_grids=None, output_grid=None, kernel=None,
                 block_padding=None, arg_cfg=None, fusable_nodes=None,
                 testing=False):
        super(StencilOclTransformer, self).__init__(
            input_grids, output_grid, kernel, arg_cfg, fusable_nodes, testing)
        self.block_padding = block_padding
        self.stencil_op = []
        self.load_mem_block = []
        self.macro_defns = []
        self.project = None
        self.boundary_kernels = None
        self.boundary_handlers = None

    def visit_Project(self, node):
        self.project = node
        node.files[0] = self.visit(node.files[0])
        return node

    def visit_CFile(self, node):
        node.body = list(map(self.visit, node.body))
        node.body.insert(0, StringTemplate("""
            #ifdef __APPLE__
            #include <OpenCL/opencl.h>
            #else
            #include <CL/cl.h>
            #endif
            """))
        return node

    def visit_FunctionDecl(self, node):
        # This function grabs the input and output grid names which are used to
        # generate the proper array macros.
        super(StencilOclTransformer, self).visit_FunctionDecl(node)
        for index, param in enumerate(node.params[:-1]):
            # TODO: Transform numpy type to ctype
            param.type = ct.POINTER(ct.c_float)()
            param.set_global()
            param.set_const()
        node.set_kernel()
        node.params[-1].set_global()
        node.params[-1].type = ct.POINTER(ct.c_float)()
        node.params.append(SymbolRef('block', ct.POINTER(ct.c_float)()))
        node.params[-1].set_local()
        node.defn = node.defn[0]

        def kernel_dim_name(cur_dim):
            return "kernel_d{}".format(cur_dim)

        def global_for_dim_name(cur_dim):
            return "global_size_d{}".format(cur_dim)

        def local_for_dim_name(cur_dim):
            return "local_size_d{}".format(cur_dim)

        # if boundary handling is copy we have to generate a collection of
        # boundary kernels to handle the on-gpu boundary copy
        if self.is_copied:
            device = cl.clGetDeviceIDs()[-1]
            self.boundary_handlers = boundary_kernel_factory(
                self.ghost_depth, self.output_grid,
                node.params[0].name,
                node.params[-2].name,  # second last parameter is output
                device
            )
            boundary_kernels = [
                FunctionDecl(
                    name=boundary_handler.kernel_name,
                    params=node.params,
                    defn=boundary_handler.generate_ocl_kernel_body(),
                )
                for boundary_handler in self.boundary_handlers
            ]

            self.project.files.append(OclFile('kernel', [node]))

            for dim, boundary_kernel in enumerate(boundary_kernels):
                boundary_kernel.set_kernel()
                self.project.files.append(OclFile(kernel_dim_name(dim), [boundary_kernel]))

            self.boundary_kernels = boundary_kernels

            # ctree.browser_show_ast(node)
            # import ctree
            # ctree.browser_show_ast(boundary_kernels[0])
        else:
            self.project.files.append(OclFile('kernel', [node]))

        # print(self.project.files[0])
        # print(self.project.files[-1])

        arg_cfg = self.arg_cfg
        if self.testing:
            local_size = (1, 1, 1)
        else:
            devices = cl.clGetDeviceIDs()
            max_sizes = cl.clGetDeviceInfo(
                devices[-1], cl.cl_device_info.CL_DEVICE_MAX_WORK_ITEM_SIZES)
            max_total = cl.clGetDeviceInfo(
                devices[-1], cl.cl_device_info.CL_DEVICE_MAX_WORK_GROUP_SIZE)

            if len(arg_cfg[0].shape) == 3:
                x_len, y_len, z_len = 1, 1, 1
                while True:
                    if arg_cfg[0].shape[0] % 2 == 1:
                        x_len = 1
                    else:
                        x_len = min(max_sizes[0], x_len * 2)
                    if max_total - z_len * x_len * y_len <= 0:
                        break
                    if arg_cfg[0].shape[1] % 2 == 1:
                        y_len = 1
                    else:
                        y_len = min(max_sizes[1], y_len * 2)
                    if max_total - z_len * x_len * y_len <= 0:
                        break
                    if arg_cfg[0].shape[2] % 2 == 1:
                        z_len = 1
                    else:
                        z_len = min(max_sizes[2], z_len * 2)
                    if max_total - z_len * x_len * y_len <= 0:
                        break
                    if x_len == arg_cfg[0].shape[0] or \
                            y_len == arg_cfg[0].shape[1] or \
                            z_len == arg_cfg[0].shape[2]:
                        break

                local_size = (x_len, y_len, z_len)
            elif len(arg_cfg[0].shape) == 2:
                x_len, y_len = 1, 1
                while True:
                    if arg_cfg[0].shape[0] % 2 == 1:
                        x_len = 1
                    else:
                        x_len = min(max_sizes[0], x_len * 2)
                    if max_total - x_len * y_len <= 0:
                        break
                    if arg_cfg[0].shape[1] % 2 == 1:
                        y_len = 1
                    else:
                        y_len = min(max_sizes[1], y_len * 2)
                    if max_total - x_len * y_len <= 0:
                        break
                    if x_len == arg_cfg[0].shape[0] or \
                            y_len == arg_cfg[0].shape[1]:
                        break

                local_size = (x_len, y_len)
                if arg_cfg[0].shape[0] % x_len != 0 or \
                   arg_cfg[0].shape[1] % y_len != 0:
                    raise StencilException(
                        'opencl backend must have sizes that are multiples of {}'.format(
                            local_size
                        ))
            else:
                local_size = (min(
                    max_total, max_sizes[0], arg_cfg[0].shape[0] / 2),)

        defn = [
            ArrayDef(
                SymbolRef('global', ct.c_ulong()), arg_cfg[0].ndim,
                [Constant(d)
                 for d in arg_cfg[0].shape]
            ),
            ArrayDef(
                SymbolRef('local', ct.c_ulong()), arg_cfg[0].ndim,
                [Constant(s) for s in local_size]
            )
        ]
        setargs = [clSetKernelArg(
            SymbolRef('kernel'), Constant(d),
            FunctionCall(SymbolRef('sizeof'), [SymbolRef('cl_mem')]),
            Ref(SymbolRef('buf%d' % d))
        ) for d in range(len(arg_cfg) + 1)]
        from functools import reduce
        import operator
        local_mem_size = reduce(
            operator.mul,
            (size + 2 * self.kernel.ghost_depth[index]
             for index, size in enumerate(local_size)),
            ct.sizeof(cl.cl_float())
        )
        setargs.append(
            clSetKernelArg(
                'kernel', len(arg_cfg) + 1,
                local_mem_size,
                NULL()
            )
        )

        defn.extend(setargs)
        enqueue_call = FunctionCall(SymbolRef('clEnqueueNDRangeKernel'), [
            SymbolRef('queue'), SymbolRef('kernel'),
            Constant(self.kernel.dim), NULL(),
            SymbolRef('global'), SymbolRef('local'),
            Constant(0), NULL(), NULL()
        ])

        defn.append(enqueue_call)

        params = [
            SymbolRef('queue', cl.cl_command_queue()),
            SymbolRef('kernel', cl.cl_kernel())
        ]
        if self.is_copied:
            for dim, boundary_kernel in enumerate(self.boundary_kernels):
                defn.extend([
                    ArrayDef(
                        SymbolRef(global_for_dim_name(dim), ct.c_ulong()), arg_cfg[0].ndim,
                        [Constant(d)
                         for d in self.boundary_handlers[dim].global_size]
                    ),
                    ArrayDef(
                        SymbolRef(local_for_dim_name(dim), ct.c_ulong()), arg_cfg[0].ndim,
                        [Constant(s) for s in self.boundary_handlers[dim].local_size]
                    )
                ])
                setargs = [clSetKernelArg(
                    SymbolRef(kernel_dim_name(dim)), Constant(d),
                    FunctionCall(SymbolRef('sizeof'), [SymbolRef('cl_mem')]),
                    Ref(SymbolRef('buf%d' % d))
                ) for d in range(len(arg_cfg) + 1)]
                setargs.append(
                    clSetKernelArg(
                        SymbolRef(kernel_dim_name(dim)), len(arg_cfg) + 1,
                        local_mem_size,
                        NULL()
                    )
                )
                defn.extend(setargs)

                enqueue_call = FunctionCall(SymbolRef('clEnqueueNDRangeKernel'), [
                    SymbolRef('queue'), SymbolRef(kernel_dim_name(dim)),
                    Constant(self.kernel.dim), NULL(),
                    SymbolRef(global_for_dim_name(dim)), SymbolRef(local_for_dim_name(dim)),
                    Constant(0), NULL(), NULL()
                ])
                defn.append(enqueue_call)

                params.extend([
                    SymbolRef(kernel_dim_name(dim), cl.cl_kernel())
                ])

        finish_call = FunctionCall(SymbolRef('clFinish'), [SymbolRef('queue')])
        defn.append(finish_call)

        params.extend(SymbolRef('buf%d' % d, cl.cl_mem())
                      for d in range(len(arg_cfg) + 1))

        control = FunctionDecl(None, "stencil_control",
                               params=params,
                               defn=defn)

        return control

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
        index = "(d%d)" % (self.output_grid.ndim - 1)
        for x in reversed(range(self.output_grid.ndim - 1)):
            ndim = str(int(self.output_grid.strides[x] /
                           self.output_grid.itemsize))
            index += "+((d%s) * %s)" % (str(x), ndim)
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
                        Constant(2 * self.ghost_depth[d])
                    ),
                ),
                point[d]
            )
        return FunctionCall(SymbolRef("local_array_macro"), point)

    def gen_local_macro(self):
        dim = len(self.output_grid.shape)
        index = SymbolRef("d%d" % (dim - 1))
        for d in reversed(range(dim - 1)):
            base = Add(get_local_size(dim - 1),
                       Constant(2 * self.ghost_depth[dim - 1]))
            for s in range(d + 1, dim - 1):
                base = Mul(
                    base,
                    Add(get_local_size(s), Constant(2 * self.ghost_depth[s]))
                )
            index = Add(
                index, Mul(base, SymbolRef("d%d" % d))
            )
            index._force_parentheses = True
            index.right.right._force_parentheses = True
        return index

    def gen_global_index(self):
        dim = self.output_grid.ndim
        index = get_global_id(dim - 1)
        for d in reversed(range(dim - 1)):
            stride = self.output_grid.strides[d] // \
                self.output_grid.itemsize
            index = Add(
                index,
                Mul(
                    get_global_id(d),
                    Constant(stride)
                )
            )
        return index

    def load_shared_memory_block(self, target, ghost_depth):
        dim = len(self.output_grid.shape)
        body = []
        thread_id, num_threads, block_size = gen_decls(dim, ghost_depth)

        body.extend([Assign(SymbolRef("thread_id", ct.c_int()), thread_id),
                     Assign(SymbolRef("block_size", ct.c_int()), block_size),
                     Assign(SymbolRef("num_threads", ct.c_int()), num_threads)
                     ])
        base = None
        for i in reversed(range(0, dim - 1)):
            if base is not None:
                base = Mul(Add(get_local_size(i + 1),
                               Constant(self.ghost_depth[i + 1] * 2)),
                           base)
            else:
                base = Add(get_local_size(i + 1),
                           Constant(self.ghost_depth[i + 1] * 2))
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
        for d in reversed(range(0, dim - 1)):
            base = None
            for i in reversed(range(d + 1, dim)):
                if base is not None:
                    base = Mul(
                        Add(get_local_size(i),
                            ghost_depth[i] * 2),
                        base
                    )
                else:
                    base = Add(get_local_size(i), Constant(ghost_depth[i] * 2))
            if base is not None and d != 0:
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
                            [FunctionCall(
                                SymbolRef('clamp'),
                                [Cast(ct.c_int(), Sub(Add(
                                    SymbolRef("local_id%d" % (dim - d - 1)),
                                    Mul(FunctionCall(
                                        SymbolRef('get_group_id'),
                                        [Constant(d)]),
                                        get_local_size(d))
                                ), Constant(self.kernel.ghost_depth[d]))),
                                    Constant(0), Constant(
                                        self.arg_cfg[0].shape[d]-1
                                    )
                                ]
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
        cond = And(
            Lt(get_global_id(0),
               Constant(self.arg_cfg[0].shape[0] - self.ghost_depth[0])),
            GtE(get_global_id(0),
                Constant(self.ghost_depth[0]))
        )
        for d in range(1, len(self.arg_cfg[0].shape)):
            cond = And(
                cond,
                And(
                    Lt(get_global_id(d),
                       Constant(self.arg_cfg[0].shape[d] - self.ghost_depth[d])),
                    GtE(get_global_id(d),
                        Constant(self.ghost_depth[d]))
                )
            )
        body = []

        self.macro_defns = [
            CppDefine("local_array_macro", ["d%d" % i for i in range(dim)],
                      self.gen_local_macro()),
            CppDefine("global_array_macro", ["d%d" % i for i in range(dim)],
                      self.gen_global_macro())
        ]
        body.extend(self.macro_defns)

        global_idx = 'global_index'
        self.output_index = global_idx
        body.append(Assign(SymbolRef('global_index', ct.c_int()),
                    self.gen_global_index()))

        self.load_mem_block = self.load_shared_memory_block(
            SymbolRef('block'), self.ghost_depth)
        body.extend(self.load_mem_block)
        body.append(FunctionCall(SymbolRef("barrier"),
                                 [SymbolRef("CLK_LOCAL_MEM_FENCE")]))
        for d in range(0, dim):
            body.append(Assign(SymbolRef('local_id%d' % d, ct.c_int()),
                               Add(get_local_id(d),
                                   Constant(self.ghost_depth[d]))))
            self.var_list.append("local_id%d" % d)

        for child in map(self.visit, node.body):
            if isinstance(child, list):
                self.stencil_op.extend(child)
            else:
                self.stencil_op.append(child)
        body.extend(self.stencil_op)
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
                    pt = list(map(lambda x: SymbolRef(x), self.var_list))
                    index = self.local_array_macro(pt)
                    return ArrayRef(SymbolRef('block'), index)
            else:
                pt = list(map(lambda x, y: Add(SymbolRef(x), Constant(y)),
                              self.var_list, self.offset_list))
                index = self.local_array_macro(pt)
                return ArrayRef(SymbolRef('block'), index)
        elif isinstance(target, FunctionCall) or \
                isinstance(target, MathFunction):
            return ArrayRef(SymbolRef(grid_name), self.visit(target))
        raise Exception(
            "Unsupported GridElement encountered: {0}".format(grid_name))


def gen_decls(dim, ghost_depth):
    thread_id = get_local_id(dim - 1)
    num_threads = get_local_size(dim - 1)
    block_size = Add(
        get_local_size(dim - 1),
        Constant(ghost_depth[dim - 1] * 2)
    )
    for d in reversed(range(0, dim - 1)):
        base = get_local_size(dim - 1)
        for s in range(d, dim - 2):
            base = Mul(get_local_size(s + 1), base)

        thread_id = Add(
            Mul(get_local_id(d), base),
            thread_id
        )
        num_threads = Mul(get_local_size(d), num_threads)
        block_size = Mul(
            Add(get_local_size(d), Constant(ghost_depth[d] * 2)),
            block_size
        )
    return thread_id, num_threads, block_size
