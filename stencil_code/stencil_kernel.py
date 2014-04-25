"""
This version was taken from the stencil_specializer project and has all asp
stuff removed in order to work on a direct c-tree llvm implementation

The main driver, intercepts the kernel() call and invokes the other components.

Stencil kernel classes are subclassed from the StencilKernel class
defined here. At initialization time, the text of the kernel() method
is parsed into a Python AST, then converted into a StencilModel by
stencil_python_front_end. The kernel() function is replaced by
shadow_kernel(), which intercepts future calls to kernel().

During each call to kernel(), stencil_unroll_neighbor_iter is called
to unroll neighbor loops, stencil_convert is invoked to convert the
model to C++, and an external compiler tool is invoked to generate a
binary which then efficiently completes executing the call. The binary
is cached for future calls.
"""

import math

from ctree.jit import LazySpecializedFunction
from ctree.transformations import FixUpParentPointers
from ctree.c.types import *
from ctree.c.nodes import *
from ctree.c.macros import *
from ctree.ocl.nodes import *
from ctree.templates.nodes import FileTemplate, StringTemplate
from ctree.frontend import get_ast
from stencil_code.backend.omp import StencilOmpTransformer
from stencil_code.backend.ocl import StencilOclTransformer
from stencil_code.backend.c import StencilCTransformer
from stencil_python_frontend import PythonToStencilModel
import stencil_optimizer as optimizer
from ctypes import byref, c_float


# logging.basicConfig(level=20)

class StencilConvert(LazySpecializedFunction):
    def __init__(self, func, input_grids, output_grid, kernel, testing=False):
        self.testing = testing
        self.input_grids = input_grids
        self.output_grid = output_grid
        self.kernel = kernel
        self.backend = kernel.backend
        if self.backend == StencilOclTransformer:
            entry_point = "stencil"
        else:
            entry_point = "stencil_kernel"
        super(StencilConvert, self).__init__(get_ast(func), entry_point)

    def args_to_subconfig(self, args):
        conf = ()
        for arg in args[:-1]:
            conf += ((len(arg), arg.dtype, arg.ndim, arg.shape),)
        return conf

    def get_tuning_driver(self):
        # if self.testing:
        from ctree.tune import (
            BruteForceTuningDriver,
            IntegerParameter,
            MinimizeTime
        )

        params = [IntegerParameter("unroll_factor", 1, 4)]
        for d in range(len(self.input_grids[0].shape) - 1):
            params.append(IntegerParameter("block_factor_%s" % d, 4, 8))
        return BruteForceTuningDriver(params, MinimizeTime())
        # else:
        #     from ctree.opentuner.driver import OpenTunerDriver
        #     from opentuner.search.manipulator import ConfigurationManipulator
        #     from opentuner.search.manipulator import PowerOfTwoParameter
        #     from opentuner.search.objective import MinimizeTime
        #
        #     manip = ConfigurationManipulator()
        #     manip.add_parameter(PowerOfTwoParameter("block_factor", 4, 8))
        #     manip.add_parameter(PowerOfTwoParameter("unroll_factor", 1, 4))
        #
        #     return OpenTunerDriver(manipulator=manip, objective=MinimizeTime())

    def transform(self, tree, program_config):
        """Convert the Python AST to a C AST."""
        param_types = []
        for arg in program_config[0]:
            param_types.append(NdPointer(arg[1], arg[2], arg[3]))
        param_types.append(Ptr(Float()))
        if self.backend == StencilOclTransformer:
            param_types.append(param_types[0])
        kernel_sig = FuncType(Void(), param_types)

        tune_cfg = program_config[1]
        block_factors = [2**tune_cfg['block_factor_%s' % d] for d in
                         range(len(self.input_grids[0].shape) - 1)]
        unroll_factor = 2**tune_cfg['unroll_factor']

        for transformer in [PythonToStencilModel(),
                            FixUpParentPointers(),
                            self.backend(self.input_grids,
                                         self.output_grid,
                                         self.kernel
                                         )]:
            tree = transformer.visit(tree)
        # first_For = tree.find(For)
        # TODO: let the optimizer handle this? Or move the find inner most loop
        # code somewhere else?
        # inner_For = optimizer.FindInnerMostLoop().find(first_For)
        # self.block(inner_For, first_For, block_factor)
        # TODO: If should unroll check
        # optimizer.unroll(inner_For, unroll_factor)
        entry_point = tree.find(FunctionDecl, name="stencil_kernel")
        entry_point.set_typesig(kernel_sig)
        # TODO: This logic should be provided by the backends
        if self.backend == StencilOclTransformer:
            entry_point.set_kernel()
            kernel = OclFile("kernel", [entry_point])
            control = CFile("control", config_target='opencl')
            body = control.body
            import os
            blk = []
            tmpl_path = os.path.join(os.getcwd(), "templates",
                                     "OclLoadGrid.tmpl.c")
            for index, param in enumerate(entry_point.params[:-1]):
                tmpl_args = {
                    'grid_size': Constant(program_config[0][index][0] **
                                          program_config[0][index][2]),
                    'arg_ref': SymbolRef(param.name),
                    'arg_index': Constant(index)
                }
                blk.append(FileTemplate(tmpl_path, tmpl_args))
            tmpl_path = os.path.join(os.getcwd(), "templates",
                                     "OclStencil.tmpl.c")
            decl = ""
            for param in entry_point.params[:-1]:
                decl += str(SymbolRef(param.name, param.type)) + ", "
            tmpl_args = {
                'use_gpu': Constant(1) if not self.kernel.testing else Constant(0),
                'array_decl': StringTemplate(decl[:-2] + ', float* duration'),
                'grid_size': Constant(program_config[0][-1][0] ** program_config[0][-1][2]),
                'kernel_path': kernel.get_generated_path_ref(),
                'kernel_name': String(entry_point.name),
                'num_args': Constant(len(entry_point.params) - 1),
                'global_size': ArrayDef([dim - 2 * self.input_grids[0].ghost_depth for dim in program_config[0][0][3]]),
                'dim': Constant(program_config[0][-1][2]),
                'output_ref': SymbolRef(entry_point.params[-2].name),
                'load_params': blk,
                'release_params': [
                    FunctionCall(
                        SymbolRef('clReleaseMemObject'),
                        ['device_' + param.name]
                    ) for param in entry_point.params[:-1]
                ]
            }
            body.append(FileTemplate(tmpl_path, tmpl_args))
            proj = Project([kernel, control])
            # from ctree.dotgen import to_dot
            # ctree.browser_show_ast(proj, 'graph.png')
            return proj, FuncType(Int(), param_types[:-1]).as_ctype()
        else:
            if self.input_grids[0].shape[len(self.input_grids[0].shape) - 1] \
                    >= unroll_factor:
                first_For = tree.find(For)
                inner_For = optimizer.FindInnerMostLoop().find(first_For)
                inner, first = optimizer.block_loops(inner_For, first_For,
                                                     block_factors + [1])
                first_For.replace(first)
                optimizer.unroll(inner, unroll_factor)

        # import ast
        #print(ast.dump(tree))
        # TODO: This should be done in the visitors
        tree.files[0].config_target = 'omp'
        return tree, entry_point.get_type().as_ctype()

    #def block(self, tree, factor):


# may want to make this inherit from something else...
class StencilKernel(object):
    backend_dict = {"c": StencilCTransformer,
                    "omp": StencilOmpTransformer,
                    "ocl": StencilOclTransformer,
                    "opencl": StencilOclTransformer}

    def __init__(self, backend="c", pure_python=False, testing=False):
        # we want to raise an exception if there is no kernel()
        # method defined.
        try:
            dir(self).index("kernel")
        except ValueError:
            raise Exception("No kernel method defined.")

        self.backend = self.backend_dict[backend]
        self.testing = testing

        self.model = self.kernel

        self.pure_python = pure_python
        self.pure_python_kernel = self.kernel
        self.should_unroll = True
        self.should_cacheblock = False
        self.block_size = 1

        # replace kernel with shadow version
        self.kernel = self.shadow_kernel

        self.specialized_sizes = None
        self.constants = {}

    def shadow_kernel(self, *args):
        if self.pure_python:
            return self.pure_python_kernel(*args)

        if not self.specialized_sizes or\
                self.specialized_sizes != [y.shape for y in args]:
            self.specialized = StencilConvert(
                self.model, args[0:-1], args[-1], self, self.testing)
            self.specialized_sizes = [arg.shape for arg in args]

        duration = c_float()
        args = [arg.data for arg in args]
        args.append(byref(duration))
        self.specialized(*args)
        self.specialized.report(time=duration)
        print("Took %.3fs" % duration.value)

    def distance(self, x, y):
        """
        default euclidean distance override this to return something
        reasonable for each neighbor cell distance
        """
        return math.sqrt(sum([(x[i]-y[i])**2 for i in range(0, len(x))]))


import time


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
