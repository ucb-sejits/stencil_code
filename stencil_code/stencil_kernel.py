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

import numpy
import math
import inspect
import ast
# from examples.stencil_grid.stencil_python_front_end import *
# from examples.stencil_grid.stencil_unroll_neighbor_iter import *
# from examples.stencil_grid.stencil_optimize_cpp import *
# from examples.stencil_grid.stencil_convert import *
from copy import deepcopy
import logging
logging.basicConfig(level=20)

from ctree.transformations import PyBasicConversions
from ctree.jit import LazySpecializedFunction
from ctree.c.types import *
from ctree.frontend import get_ast
from ctree.visitors import NodeTransformer, NodeVisitor
from ctree.c.nodes import *
from ctree.omp.nodes import *



# logging.basicConfig(filename='tmp.txt',
#                             filemode='w',
#                             format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#                             datefmt='%H:%M:%S',
#                             level=20)


class StencilConvert(LazySpecializedFunction):
    def __init__(self, func, entry_point, input_grids, output_grid, constants):
        self.input_grids = input_grids
        self.output_grid = output_grid
        self.constants = constants
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

        for transformer in [StencilTransformer(self.input_grids,
                                               self.output_grid,
                                               self.constants
                                               ),
                            PyBasicConversions()]:
            tree = transformer.visit(tree)
        first_For = tree.find(For)
        inner_For = FindInnerMostLoop().find(first_For)
        # self.block(inner_For, first_For, block_factor)
        self.unroll(inner_For, unroll_factor)
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

    def unroll(self, for_node, factor):
        # Determine the leftover iterations after unrolling
        initial = for_node.init.right.value
        end = for_node.test.right.value
        leftover_begin = int((end - initial + 1) / factor) * factor + initial

        new_end = leftover_begin - 1
        new_incr = AddAssign(SymbolRef(for_node.incr.arg.name), factor)
        new_body = for_node.body[:]
        for x in range(1, factor):
            new_extension = deepcopy(for_node.body)
            new_extension = map(UnrollReplacer(for_node.init.left.name,
                                               x).visit, new_extension)
            new_body.extend(new_extension)

        leftover_For = For(Assign(for_node.init.left,
                                  Constant(leftover_begin)),
                           for_node.test,
                           for_node.incr,
                           for_node.body)
        for_node.test = LtE(for_node.init.left.name, new_end)
        for_node.incr = new_incr
        for_node.body = new_body

        if not leftover_begin >= end:
            for_node.body.append(leftover_For)

    #def block(self, tree, factor):



class FindInnerMostLoop(NodeVisitor):
    def __init__(self):
        self.inner_most = None

    def find(self, node):
        self.visit(node)
        return self.inner_most

    def visit_For(self, node):
        self.inner_most = node
        map(self.visit, node.body)


class UnrollReplacer(NodeTransformer):
    def __init__(self, loopvar, incr):
        self.loopvar = loopvar
        self.incr = incr
        self.in_new_scope = False
        self.inside_for = False
        super(UnrollReplacer, self).__init__()

    def visit_SymbolRef(self, node):
        if node.name == self.loopvar:
            return Add(node, Constant(self.incr))
        return SymbolRef(node.name)


class StencilTransformer(NodeTransformer):
    def __init__(self, input_grids, output_grid, constants):
        # TODO: Give these wrapper classes?
        self.input_grids = input_grids
        self.output_grid = output_grid
        self.ghost_depth = output_grid.ghost_depth
        self.next_fresh_var = 0
        self.output_index = None
        self.neighbor_grid_name = None
        self.kernel_target = None
        self.offset_list = None
        self.var_list = []
        self.input_dict = {}
        self.constants = constants
        super(StencilTransformer, self).__init__()

    def visit_FunctionDef(self, node):
        for index, arg in enumerate(node.args.args[1:]):
            # PYTHON3 vs PYTHON2
            if hasattr(arg, 'arg'):
                arg = arg.arg
            else:
                arg = arg.id
            if index < len(self.input_grids):
                self.input_dict[arg] = self.input_grids[index]
            else:
                self.output_grid_name = arg
        node.body = list(map(self.visit, node.body))
        return node

    def gen_fresh_var(self):
        self.next_fresh_var += 1
        return "x%d" % self.next_fresh_var

    def visit_For(self, node):
        if type(node.iter) is ast.Call and \
           type(node.iter.func) is ast.Attribute:
            if node.iter.func.attr is 'interior_points':
                dim = len(self.output_grid.shape)
                self.kernel_target = node.target.id
                curr_node = None
                ret_node = None
                for d in range(dim):
                    target = SymbolRef(self.gen_fresh_var())
                    self.var_list.append(target.name)
                    for_loop = For(
                        Assign(SymbolRef(target.name, Int()),
                               Constant(self.ghost_depth)),
                        LtE(target,
                            Constant(
                                self.output_grid.shape[d] -
                                self.ghost_depth - 1)
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
                curr_node.body = [Assign(SymbolRef(self.output_index, Int()),
                                         macro)]
                for elem in map(self.visit, node.body):
                    if type(elem) == list:
                        curr_node.body.extend(elem)
                    else:
                        curr_node.body.append(elem)
                self.kernel_target = None
                return ret_node
            elif node.iter.func.attr is 'neighbors':
                neighbors_id = node.iter.args[1].n
                grid_name = node.iter.func.value.id
                grid = self.input_dict[grid_name]
                zero_point = tuple([0 for x in range(grid.dim)])
                self.neighbor_target = node.target.id
                self.neighbor_grid_name = grid_name
                body = []
                statement = node.body[0]
                for x in grid.neighbors(zero_point, neighbors_id):
                    self.offset_list = list(x)
                    for statement in node.body:
                        body.append(self.visit(deepcopy(statement)))
                self.neighbor_target = None
                return body
        return node

    # Handle array references
    def visit_Subscript(self, node):
        grid_name = node.value.id
        target = node.slice.value
        if isinstance(target, ast.Name):
            target = target.id
            if target == self.kernel_target:
                if grid_name is self.output_grid_name:
                    return ArrayRef(SymbolRef(self.output_grid_name),
                                    SymbolRef(self.output_index))
                elif grid_name in self.input_dict:
                    grid = self.input_dict[grid_name]
                    pt = list(map(lambda x: SymbolRef(x), self.var_list))
                    index = self.gen_array_macro(grid_name, pt)
                    return ArrayRef(SymbolRef(grid_name), index)
            elif grid_name == self.neighbor_grid_name:
                pt = list(map(lambda x, y: Add(SymbolRef(x), SymbolRef(y)),
                              self.var_list, self.offset_list))
                index = self.gen_array_macro(grid_name, pt)
                return ArrayRef(SymbolRef(grid_name), index)
        elif isinstance(target, ast.Call):
            return ArrayRef(SymbolRef(grid_name), self.visit(target))
        return node

    def visit_Call(self, node):
        if node.func.id == 'distance':
            zero_point = tuple([0 for _ in range(len(self.offset_list))])
            return Constant(int(self.distance(zero_point, self.offset_list)))
        elif node.func.id == 'int':
            return Cast(Int(), self.visit(node.args[0]))
        node.args = list(map(self.visit, node.args))
        return node

    def distance(self, x, y):
        return math.sqrt(sum([(x[i]-y[i])**2 for i in range(0, len(x))]))

    def gen_array_macro(self, arg, point):
        name = "_%s_array_macro" % arg
        return FunctionCall(SymbolRef(name), point)

    def visit_AugAssign(self, node):
        # TODO: Handle all types?
        value = self.visit(node.value)
        # HACK to get this to work, PyBasicConversions will skip this AugAssign node
        # TODO Figure out why
        value = PyBasicConversions().visit(value)
        if type(node.op) is ast.Add:
            return AddAssign(self.visit(node.target), value)
        if type(node.op) is ast.Sub:
            return SubAssign(self.visit(node.target), value)

    def visit_Assign(self, node):
        target = PyBasicConversions().visit(self.visit(node.targets[0]))
        value = PyBasicConversions().visit(self.visit(node.value))
        return Assign(target, value)

    def visit_Name(self, node):
        if node.id in self.constants.keys():
            return Constant(self.constants[node.id])
        raise Exception("Undeclared name %s. \
                Please add it to your kernel's self.constants" % node.id)
        return node


# may want to make this inherit from something else...
class StencilKernel(object):
    def __init__(self, with_cilk=False):
        # we want to raise an exception if there is no kernel()
        # method defined.
        try:
            dir(self).index("kernel")
        except ValueError:
            raise Exception("No kernel method defined.")

        # get text of kernel() method and parse into a StencilModel
        # self.kernel_src = inspect.getsource(self.kernel)
        # print(self.kernel_src)
        # self.kernel_ast = ast.parse(self.remove_indentation(self.kernel_src))
        # print(ast.dump(self.kernel_ast, include_attributes=True))

        # self.model = StencilPythonFrontEnd().parse(self.kernel_ast)
        # print(ast.dump(self.model, include_attributes=True))

        self.model = self.kernel
        # print(self.new_kernel)

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
                self.model, "kernel", args[0:-1], args[-1], self.constants)
            self.specialized_sizes = [arg.shape for arg in args]

        with Timer() as t:
            self.specialized(*[arg.data for arg in args])
        self.specialized.report(time=t)

import time


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
