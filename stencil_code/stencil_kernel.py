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

from ctree.transformations import PyBasicConversions
from ctree.jit import LazySpecializedFunction
from ctree.c.types import *
from ctree.c.nodes import *
from ctree.frontend import get_ast

import stencil_optimizer as optimizer
from stencil_omp_transformer import StencilOmpTransformer


class StencilConvert(LazySpecializedFunction):
    def __init__(self, func, entry_point, input_grids, output_grid, kernel):
        self.input_grids = input_grids
        self.output_grid = output_grid
        self.kernel = kernel
        super(StencilConvert, self).__init__(get_ast(func), entry_point)

    def args_to_subconfig(self, args):
        conf = ()
        for arg in args:
            conf += ((len(arg), arg.dtype, arg.ndim, arg.shape),)
        return conf

    def get_tuning_driver(self):
        from ctree.tune import (
            BruteForceTuningDriver,
            IntegerParameter,
            MinimizeTime
        )

        params = [IntegerParameter("block_factor", 4, 8),
                  IntegerParameter("unroll_factor", 1, 4)]
        return BruteForceTuningDriver(params, MinimizeTime())

    def transform(self, tree, program_config):
        """Convert the Python AST to a C AST."""
        param_types = []
        for arg in program_config[0]:
            param_types.append(NdPointer(arg[1], arg[2], arg[3]))
        kernel_sig = FuncType(Void(), param_types)

        tune_cfg = program_config[1]
        # block_factor = 2**tune_cfg['block_factor']
        unroll_factor = 2**tune_cfg['unroll_factor']

        for transformer in [StencilOmpTransformer(self.input_grids,
                                                  self.output_grid,
                                                  self.kernel
                                                  ),
                            PyBasicConversions()]:
            tree = transformer.visit(tree)
        first_For = tree.find(For)
        # TODO: let the optimizer handle this? Or move the find inner most loop
        # code somewhere else?
        inner_For = optimizer.FindInnerMostLoop().find(first_For)
        # self.block(inner_For, first_For, block_factor)
        # TODO: If should unroll check
        optimizer.unroll(inner_For, unroll_factor)
        # remove self param
        # TODO: Better way to do this?
        params = tree.find(FunctionDecl, name="kernel").params
        params.pop(0)
        self.gen_array_macro_definition(tree, params)
        entry_point = tree.find(FunctionDecl, name="kernel")
        entry_point.set_typesig(kernel_sig)
        return tree, entry_point.get_type().as_ctype()

    def gen_array_macro_definition(self, tree, arg_names):
        first_for = tree.find(For)
        for index, arg in enumerate(self.input_grids + (self.output_grid,)):
            defname = "_%s_array_macro" % arg_names[index]
            params = ','.join(["_d"+str(x) for x in range(arg.dim)])
            params = "(%s)" % params
            calc = "((_d%d)" % (arg.dim - 1)
            for x in range(arg.dim - 1):
                dim = str(int(arg.data.strides[x]/arg.data.itemsize))
                calc += "+((_d%s) * %s)" % (str(x), dim)
            calc += ")"
            first_for.insert_before(Define(defname+params, calc))


    #def block(self, tree, factor):


# may want to make this inherit from something else...
class StencilKernel(object):
    def __init__(self, with_cilk=False):
        # we want to raise an exception if there is no kernel()
        # method defined.
        try:
            dir(self).index("kernel")
        except ValueError:
            raise Exception("No kernel method defined.")

        self.model = self.kernel

        self.pure_python = False
        self.pure_python_kernel = self.kernel
        self.should_unroll = True
        self.should_cacheblock = False
        self.block_size = 1

        # replace kernel with shadow version
        self.kernel = self.shadow_kernel

        self.specialized_sizes = None
        self.with_cilk = with_cilk
        self.constants = {}

    def shadow_kernel(self, *args):
        if self.pure_python:
            return self.pure_python_kernel(*args)

        if not self.specialized_sizes or\
                self.specialized_sizes != [y.shape for y in args]:
            self.specialized = StencilConvert(
                self.model, "kernel", args[0:-1], args[-1], self)
            self.specialized_sizes = [arg.shape for arg in args]

        with Timer() as t:
            self.specialized(*[arg.data for arg in args])
        self.specialized.report(time=t)

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
