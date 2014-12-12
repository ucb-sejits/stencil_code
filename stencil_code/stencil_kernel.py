"""
This version was taken from the stencil_specializer project and has all asp
stuff removed in order to work on a direct c-tree llvm implementation

The main driver, intercepts the kernel() call and invokes the other components.

Stencil kernel classes are sub-classed from the StencilKernel class
defined here. At initialization time, the text of the kernel() method
is parsed into a Python AST, then converted into a StencilModel by
stencil_python_front_end.

During each call to kernel(), stencil_unroll_neighbor_iter is called
to unroll neighbor loops, stencil_convert is invoked to convert the
model to C++, and an external compiler tool is invoked to generate a
binary which then efficiently completes executing the call. The binary
is cached for future calls.
"""
from __future__ import print_function
import math

from collections import namedtuple
from numpy import zeros

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.c.nodes import FunctionDecl
from ctree.ocl.nodes import OclFile
import ctree.np
from stencil_code.stencil_exception import StencilException

_ = ctree.np  # Make PEP8 happy, and pycharm
import ctree.ocl
from ctree.ocl import get_context_and_queue_from_devices
from ctree.frontend import get_ast
from .backend.omp import StencilOmpTransformer
from .backend.ocl import StencilOclTransformer
from .backend.c import StencilCTransformer
from .python_frontend import PythonToStencilModel
# import optimizer as optimizer
from ctypes import byref, c_float, CFUNCTYPE, c_void_p, POINTER
import pycl as cl
from pycl import (
    clCreateProgramWithSource, buffer_from_ndarray, buffer_to_ndarray, cl_mem
)
import numpy as np
import ast
import itertools
import abc

from stencil_code.halo_enumerator import HaloEnumerator


def product(nums):
    result = 1
    for x in nums:
        result *= x
    return result


class ConcreteStencil(ConcreteSpecializedFunction):
    """StencilFunction

    The standard concrete specialized function that is returned when using the
    C or OpenMP backend.
    """

    def finalize(self, tree, entry_name, entry_type, output):
        """

        :param tree: A project node containing any files to be compiled for
                     this specialized function.
        :type tree: Project node
        :param entry_name: The name of the function that will be the entry
                           point to the compiled project.
        :type: str
        :param entry_type: The type signature of the function described by
                           `entry_name`.
        :type entry_type: CFUNCTYPE
        :param output: the stencil result buffer
        :return:
        """
        self.output = output
        self._c_function = self._compile(entry_name, tree, entry_type)
        return self

    def __call__(self, *args):
        """__call__

        :param *args: Arguments to be passed to our C function, the types
                      should match the types specified by the `entry_type`
                      that was passed to :attr: `finalize`.

        """
        # TODO: provide stronger type checking to give users better error messages.
        duration = c_float()
        if self.output is not None:
            output = self.output
            self.output = None
        else:  # pragma no cover
            output = np.zeros_like(args[0])
        args += (output, byref(duration))
        self._c_function(*args)
        return output


class OclStencilFunction(ConcreteSpecializedFunction):
    """OclStencilFunction

    The ConcreteSpecializedFunction used by the OpenCL backend.  Allows us to
    leverage pycl for handling numpy arrays and buffers cleanly.
    """
    def __init__(self):
        """__init__
        Creates a context and queue that can be reused across calls to this
        function.
        """
        # TODO: Need dependency injection to control ocl device selection
        self.desired_ocl_device = -1
        devices = cl.clGetDeviceIDs()
        self.context, self.queue = get_context_and_queue_from_devices([devices[self.desired_ocl_device]])
        self.max_work_group_size = devices[self.desired_ocl_device].max_work_group_size

        # some variables that will be used that PEP-8 wants to see initialized in __init__
        self.kernel = None
        self.output = None
        self._c_function = None

    def finalize(self, tree, entry_type, entry_name, kernel, output_grid):
        """
        finalize
        :param tree: the transformed tree
        :param entry_type:
        :param entry_name:
        :param kernel: the kernel generated that will be used in __call__
        :param output_grid:
        :return: a specialized function
        """
        self.kernel = kernel
        self.output = output_grid
        self._c_function = self._compile(entry_name, tree, entry_type)
        return self

    def __call__(self, *args):
        """__call__

        :param *args:
        """
        if self.output is not None:
            output = self.output
            self.output = None
        else:  # pragma no cover
            output = np.zeros_like(args[0])
        # self.kernel.argtypes = tuple(
        #     cl_mem for _ in args + (output, )
        # ) + (localmem, )
        buffers = []
        events = []
        for index, arg in enumerate(args + (output, )):
            buf, evt = buffer_from_ndarray(self.queue, arg, blocking=True)
            # evt.wait()
            events.append(evt)
            buffers.append(buf)
            # self.kernel.setarg(index, buf, sizeof(cl_mem))
        cl.clWaitForEvents(*events)
        if isinstance(self.kernel, list):
            kernels = len(self.kernel)
            if kernels == 2:
                self._c_function(self.queue, self.kernel[0], self.kernel[1], *buffers)
            if kernels == 3:
                self._c_function(self.queue, self.kernel[0], self.kernel[1], self.kernel[2], *buffers)
            if kernels == 4:
                self._c_function(self.queue, self.kernel[0], self.kernel[1], self.kernel[2], self.kernel[3], *buffers)
        else:
            self._c_function(self.queue, self.kernel,
                             *buffers)

        buf, evt = buffer_to_ndarray(
            self.queue, buffers[-1], output
        )
        evt.wait()

        return buf

    def __del__(self):
        del self.context
        del self.queue


StencilArgConfig = namedtuple(
    'StencilArgConfig', ['size', 'dtype', 'ndim', 'shape']
)


class SpecializedStencil(LazySpecializedFunction):
    backend_dict = {"c": StencilCTransformer,
                    "omp": StencilOmpTransformer,
                    "ocl": StencilOclTransformer,
                    "opencl": StencilOclTransformer}

    def __init__(self, stencil_kernel, backend, ):
        """
        Initializes an instance of a SpecializedStencil. This function
        inherits from ctree's LazySpecializedFunction. When the specialized
        function is called, it will either load a cached version, or generate
        a new version using the kernel method's AST and the passed parameters
        . The tuning configurations are defined in get_tuning_driver. The
        arguments to the specialized function call are passed to
        args_to_subconfig where they can be processed to a form usable by the
        specializer. For more information consult the ctree docs.

        :param stencil_kernel: The Kernel object containing the kernel function.
        :param backend: the type of specialized kernel to generate
\        """
        self.kernel = stencil_kernel
        self.backend = self.backend_dict[backend]
        self.output = None
        self.args = None
        super(SpecializedStencil, self).__init__(get_ast(stencil_kernel.kernel))

    def args_to_subconfig(self, args):
        """
        Generates a configuration for the transform method based on the
        arguments passed into the stencil.

        :param args: StencilGrid instances being passed as params.
        :return: Tuple of information about the StencilGrids
        """
        self.args = args
        return tuple(
            StencilArgConfig(len(arg), arg.dtype, arg.ndim, arg.shape)
            for arg in args
        )

    # def get_tuning_driver(self):
    #     """
    #     Returns the tuning driver used for this Specialized Function.
    #     Initializes a brute force tuning driver that explores the space of
    #     loop unrolling factors as well as cache blocking factors for each
    #     dimension of our input StencilGrids.

    #     :return: A BruteForceTuning driver instance
    #     """
    #     from ctree.tune import (
    #         BruteForceTuningDriver,
    #         IntegerParameter,
    #         MinimizeTime
    #     )

    #     params = [IntegerParameter("unroll_factor", 1, 4)]
    #     for d in range(len(self.input_grids[0].shape) - 1):
    #         params.append(IntegerParameter("block_factor_%s" % d, 4, 8))
    #     return BruteForceTuningDriver(params, MinimizeTime())

    def transform(self, tree, program_config):
        """
        Transforms the python AST representing our un-specialized stencil
        kernel into a c_ast which can be JIT compiled.

        :param tree: python AST of the kernel method.
        :param program_config: The configuration generated by args_to_subconfig
        :return: A ctree Project node, and our entry point type signature.
        """
        arg_cfg, tune_cfg = program_config
        output = self.generate_output(program_config)
        param_types = [
            np.ctypeslib.ndpointer(arg.dtype, arg.ndim, arg.shape)
            for arg in arg_cfg + (output,)
        ]

        if self.backend == StencilOclTransformer:
            param_types.append(param_types[0])
        else:
            param_types.append(POINTER(c_float))

        # block_factors = [2**tune_cfg['block_factor_%s' % d] for d in
        #                  range(len(self.input_grids[0].shape) - 1)]
        # unroll_factor = 2**tune_cfg['unroll_factor']
        unroll_factor = 0

        for transformer in [
            PythonToStencilModel(),
            self.backend(self.args, output, self.kernel, arg_cfg=arg_cfg,
                         fusable_nodes=None)
        ]:
            tree = transformer.visit(tree)

        # first_For = tree.find(For)
        # TODO: let the optimizer handle this? Or move the find inner most loop
        # code somewhere else?
        # inner_For = optimizer.FindInnerMostLoop().find(first_For)
        # self.block(inner_For, first_For, block_factor)
        # TODO: If should unroll check
        # optimizer.unroll(inner_For, unroll_factor)
        entry_point = tree.find(FunctionDecl, name="stencil_kernel")
        # TODO: This should be handled by the backend
        # if self.backend != StencilOclTransformer:
        for index, _type in enumerate(param_types):
            entry_point.params[index].type = _type()
        # entry_point.set_typesig(kernel_sig)
        # TODO: This logic should be provided by the backends
        if self.backend == StencilOclTransformer:
            entry_point = 'stencil_control'
            entry_type = [None, cl.cl_command_queue, cl.cl_kernel]
            if self.kernel.is_copied:
                for _ in range(self.kernel.dim):
                    entry_type.append(cl.cl_kernel)
            entry_type.extend(cl_mem for _ in range(len(arg_cfg) + 1))
            entry_type = CFUNCTYPE(*entry_type)
            return tree, entry_type, entry_point
        else:
            if self.args[0].shape[len(self.args[0].shape) - 1] \
                    >= unroll_factor:
                # FIXME: Lack of parent pointers breaks current loop unrolling
                # first_For = tree.find(For)
                # inner_For = optimizer.FindInnerMostLoop().find(first_For)
                # inner, first = optimizer.block_loops(inner_For, tree,
                #                                      block_factors + [1])
                # first_For.replace(first)
                # optimizer.unroll(tree, inner_For, unroll_factor)
                pass

        # import ast
        # print(ast.dump(tree))
        # TODO: This should be done in the visitors
        tree.files[0].config_target = 'omp'
        param_types = [
            np.ctypeslib.ndpointer(arg.dtype, arg.ndim, arg.shape)
            for arg in arg_cfg + (self.output, )
        ]
        entry_point = 'stencil_kernel'
        param_types.append(POINTER(c_float))
        entry_type = CFUNCTYPE(c_void_p, *param_types)
        return tree, entry_type, entry_point

    def finalize(self, tree, entry_type, entry_point):
        if self.backend == StencilOclTransformer:
            fn = OclStencilFunction()
            if self.kernel.is_copied:
                args = [
                    tree, entry_type, entry_point,
                ]
                kernels = []
                for index, kernel in enumerate(tree.find_all(OclFile)):
                    # print("XXX index {} kernel {}".format(index, kernel.name))
                    # print(kernel.codegen())
                    program = clCreateProgramWithSource(fn.context, kernel.codegen()).build()
                    ocl_kernel_name = 'stencil_kernel' if index == 0 else kernel.name
                    kernel_ptr = program[ocl_kernel_name]
                    kernels.append(kernel_ptr)
                args.append(kernels)
                args.append(self.output)

                finalized = fn.finalize(*args)
            else:
                kernel = tree.find(OclFile)
                program = clCreateProgramWithSource(fn.context,
                                                    kernel.codegen()).build()
                stencil_kernel_ptr = program['stencil_kernel']
                finalized = fn.finalize(
                    tree, entry_type, entry_point,
                    stencil_kernel_ptr,
                    self.output
                )
        else:
            fn = ConcreteStencil()
            finalized = fn.finalize(tree, entry_point, entry_type,
                                    self.output)
        self.output = None
        self.fusable_nodes = []
        return finalized

    def generate_output(self, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        if self.output is None:
            self.output = zeros(arg_cfg[0].shape, arg_cfg[0].dtype)
        return self.output


class Stencil(object):
    """
    Stencil is an abstract class that requires
    a kernel method
    if neighborhoods is not passed to __init__ it may be defined as
    a class level variable in the subclass
    """
    backend_dict = {"c": StencilCTransformer,
                    "omp": StencilOmpTransformer,
                    "ocl": StencilOclTransformer,
                    "opencl": StencilOclTransformer,
                    "python": None}

    boundary_handling_list = ['clamp', 'zero', 'copy', 'wrap']

    def __call__(self, *args, **kwargs):
        return self.specializer(*args, **kwargs)

    def __init__(self, backend='ocl', neighborhoods=None, boundary_handling='clamp', **kwargs):
        """
        Our Stencil class wraps an un-specialized stencil kernel
        function.  This class should be sub-classed by the user, and should
        have a kernel method defined.  When initialized, an instance of
        StencilKernel will store the kernel method and replace it with a
        lazy specialized class, which when called will begin the JIT
        specialization process using ctree's infrastructure.

        :param backend: Optional backend that should be used by ctree.
        Supported backends are c, omp (openmp), and ocl (opencl).
        :param neighborhood_definition: an iterable of neighborhoods neighborhoods are a list of points(tuples)
        :param boundary_handling: one of skip, clamped, copy; default is clamped
        :raise Exception: If no kernel method is defined.
        """

        if neighborhoods:
            self.neighborhood_definition = neighborhoods
        else:
            try:
                # self.neighborhoods below actually references the subclass variable
                self.neighborhood_definition = self.neighborhoods
            except Exception:
                raise StencilException(
                    "Error: neighborhoods must be defined by {}".format(type(self))
                )

        self.backend = self.backend_dict[backend]

        if not boundary_handling in Stencil.boundary_handling_list:
            raise StencilException("Error: boundary handling value '{}' not recognized".format(boundary_handling))

        self.boundary_handling = boundary_handling
        self.is_clamped = boundary_handling == 'clamp'
        self.is_warped = boundary_handling == 'warp'
        self.is_copied = boundary_handling == 'copy'
        self.is_zeroed = boundary_handling == 'zero'

        self.current_shape = None  # this is used to communicate shape info from interior points to neighbors

        try:
            self.dim = len(self.neighborhood_definition[0][0])
        except Exception:
            raise StencilException(
                "Error: neighborhoods not properly set for {}".format(type(self))
            )

        self.ghost_depth = self.compute_ghost_depth()

        if backend == 'python':
            self.specializer = self.python_kernel_wrapper
        elif backend in ['c', 'omp', 'ocl']:
            self.specializer = SpecializedStencil(self, backend,)
        self.model = self.kernel

        self.should_unroll = kwargs.get('should_unroll', True)
        self.should_cacheblock = kwargs.get('should_cacheblock', False)
        self.block_size = kwargs.get('block_size', 1)

        self.specialized_sizes = None

    @abc.abstractmethod
    def kernel(self, *args):
        """subclasses must implement this"""
        return

    def python_kernel_wrapper(self, *args):
        """
        create an output buffer based on input_buffer then call the kernel
        :param args:
        :return:
        """
        input_grid = args[0]
        output = np.zeros_like(input_grid)
        self.kernel(*(args + (output,)))

        if self.is_copied:
            for point in self.halo_points(input_grid):
                output[point] = input_grid[point]

        return output

    @property
    def constants(self):
        return {}

    def compute_ghost_depth(self):
        """
        figure out a maximal ghost depth for all neighborhoods
        The ghost depth may be asymmetric and is therefore a tuple of the ghost depth for each dimension
        :return: the maximal_ghost_depth tuple
        """
        ghost_depth = tuple(0 for _ in range(self.dim))
        for neighborhood in self.neighborhood_definition:
            for neighbor in neighborhood:
                ghost_depth = tuple(
                    max(ghost_depth[i], abs(neighbor[i]))
                    for i in range(self.dim)
                )
        return ghost_depth

    def interior_points(self, x):
        """
        an iterator over the points in a matrix being operated on.  The behaviour
        of this method depends on the boundary_handling
        DANGER: clamping requires that the neighbors method have access to the current shape,
                which means this functions is not re-entrant, it cannot be called from
                separate threads with different shapes
        :param x: the matrix to iterate over, typically this is the output matrix
        :return:
        """
        if self.is_clamped:
            self.current_shape = x.shape
            dims = (range(0, dim) for dim in x.shape)
        elif self.is_copied:
            dims = (range(self.ghost_depth[index], dim - self.ghost_depth[index])
                    for index, dim in enumerate(x.shape))
        else:
            dims = (range(self.ghost_depth[index], dim - self.ghost_depth[index])
                    for index, dim in enumerate(x.shape))

        for item in itertools.product(*dims):
            yield tuple(item)

    def neighbors(self, point, neighbors_id=0):
        """
        iterate over the neighborhood of point
        :param point: the nominal center of the neighborhood, neighborhood does not have to be symmetric
        :param neighbors_id: users may define more than one neighborhood
        :return: yields absolute neighbor point
        """
        try:
            if self.is_clamped and self.current_shape is not None:
                for neighbor in self.neighborhood_definition[neighbors_id]:
                    yield tuple(map(
                        lambda dim: Stencil.clamp(point[dim]+neighbor[dim], 0, self.current_shape[dim]),
                        range(len(point))))
            else:
                for neighbor in self.neighborhood_definition[neighbors_id]:
                    yield tuple(map(lambda a, b: a+b, list(point), list(neighbor)))

        except IndexError:
            raise StencilException(
                "Undefined neighborhood identifier {} this stencil has {}".format(
                    neighbors_id, len(self.neighborhood_definition)))

    def interior_points_slice(self, extra=0):
        """
        this is a convenience method that returns a tuple of the form
        (x:-x, ...) because areas of ghost zone may not be calculated, this can
        be used to limit the region of a test like
        :param extra: in case you want to further decrease the interior side
        :return: a tuple representing a numpy slice that selects the interior of a grid
        """
        return tuple([slice(x+extra, -(x+extra)) for x in self.ghost_depth])

    def distance(self, x, y):
        """
        default euclidean distance override this to return something
        reasonable for each neighbor cell distance
        :param x: Point represented as a list or tuple
        :param y: Point represented as a list or tuple
        """
        return math.sqrt(sum([(x[i]-y[i])**2 for i in range(0, len(x))]))

    @staticmethod
    def clamp(x, min_x, max_x):
        return max(min_x, min(x, max_x-1))

    def halo_points(self, grid):
        """
        generator for points not in the interior of the specified grid based on this stencil's
        ghost_depth and boundary handling
        :param grid:
        :return:
        """
        for halo_point in HaloEnumerator(self.ghost_depth, grid.shape):
            yield halo_point

