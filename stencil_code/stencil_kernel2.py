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
from benchmarks.stencil_numpy import numpy

import ctree.np

ctree.np  # Make PEP8 happy
import ctree.ocl
from ctree.frontend import get_ast
from .backend.omp import StencilOmpTransformer
from .backend.ocl import StencilOclTransformer, StencilOclSemanticTransformer
from .backend.c import StencilCTransformer
from .python_frontend import PythonToStencilModel
from ctypes import byref, c_float, CFUNCTYPE, c_void_p, POINTER
import numpy as np
import itertools
from stencil_code.stencil_kernel import SpecializedStencil


class StencilKernel2(object):
    """
    Is an abstract stencil operator
    requires
    kernel iteration implemenation
    allows
    neighborhood(s) specification
    coefficient specification
    boundary handling specification
    """
    backend_dict = {"c": StencilCTransformer,
                    "omp": StencilOmpTransformer,
                    "ocl": StencilOclTransformer,
                    "opencl": StencilOclTransformer,
                    "python": None}

    boundary_handling_methods = {'clamped', 'none', 'toroid'}

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, value):
        if value not in StencilKernel2.backend_dict:
            raise "Unsupported backend {} assigned to this StencilKernel".format(value)
        self._backend = value
        self.handle_configuration_change()

    OPENCL = 1
    OPENMP = 2
    BEST = 3

    CLAMPED = 1
    WRAP=2
    class CONSTANT(object):
        def __init__(self, value):
            self.value = value

    s = StencilKernel2(boundary_handling_methods=StencilKernel2.CONSTANT(6.0))

    def __init__(self, neighborhood_definition=None, coefficient_definition=None,
                 backend=StencilKernel2.BEST, boundary_handling=None, testing=False):
        """
        Our StencilKernel class wraps an un-specialized stencil kernel
        function.  This class should be sub-classed by the user, and should
        have a kernel method defined.  When initialized, an instance of
        StencilKernel will store the kernel method and replace it with a
        shadow_kernel method, which when called will begin the JIT
        specialization process using ctree's infrastructure.

        :param backend: Optional backend that should be used by ctree.
        Supported backends are c, omp (openmp), and ocl (opencl).
        :param pure_python: Setting this will true will cause the python
        version of the kernel to be preserved.  Any subsequent calls will be
        run in python without any JIT specializiation.
        :param testing: Used for testing.
        :raise Exception: If no kernel method is defined.
        """

        # we want to raise an exception if there is no kernel()
        # method defined.
        try:
            dir(self).index("kernel")
        except ValueError:
            raise Exception("No kernel method defined.")

        self._backend = None    # for PEP8
        self.backend = backend  # this will kick off necessary backend checking and handling
        self.boundary_handling = boundary_handling
        self.neighborhoods = []
        self.coefficients = None
        self.dim = 2

        self.testing = testing

        self.model = self.kernel

        # self.pure_python = pure_python
        # self.pure_python_kernel = self.kernel
        self.should_unroll = True
        self.should_cacheblock = False
        self.block_size = 1

        # replace kernel with shadow version
        # self.kernel = self.shadow_kernel

        self.specialized_sizes = None

    def handle_configuration_change(cls, backend="c", testing=False):
        cls.dim = len(cls.neighbor_definition[0][0])
        ghost_depth = tuple(0 for _ in range(cls.dim))
        for neighborhood in cls.neighbor_definition:
            for neighbor in neighborhood:
                ghost_depth = tuple(
                    max(ghost_depth[i], abs(neighbor[i]))
                    for i in range(cls.dim)
                )
        cls.ghost_depth = ghost_depth
        if backend == 'python':
            cls.__call__ = cls.pure_python
            return super(StencilKernel2, cls).__new__(cls)
        elif backend in ['c', 'omp', 'ocl']:
            new = super(StencilKernel2, cls).__new__(cls)
            return SpecializedStencil(new, backend, testing)

    def parse_neighborhoods(self, neighborhood_list):
        def add_neighborhood(neighbor_hood):
            self.neighborhoods.append(neighbor_hood)

        self.neighborhoods = []
        if isinstance(neighborhood_list, list):
            if isinstance(neighborhood_list[0], list):
                map(add_neighborhood, list)
            else:
                add_neighborhood(list)
        else:
            raise "Invalid neighborhood specifier {}, must be a list".format()

    def parse_coefficients(self, coefficient_specification):
        coefficient_specification = numpy.array(coefficient_specification)
        self.dim = len(coefficient_specification.shape)

        neighbor_list = []


    def pure_python(self, *args):
        output = np.zeros_like(args[0])
        self.kernel(*(args + (output,)))
        return output

    @property
    def constants(self):
        return {}

    def shadow_kernel(self, *args):
        """
        This shadow_kernel method will replace the kernel method that is
        defined in the sub-class of StencilKernel.  If in pure python mode,
        it will execute the kernel in python.  Else, it first checks if we
        have a cached version of the specialized function for the shapes of
        the arguments.  If so, we make a call to that function with our new
        arguments.  If not, we create a new SpecializedStencil with our
        arguments and original kernel method and call it with our arguments.
        :param args: The arguments to our original kernel method.
        :return: Undefined
        """
        output_grid = np.zeros_like(args[0])
        # output_grid = StencilGrid(args[0].shape)
        # output_grid.ghost_depth = self.ghost_depth
        if self.pure_python:
            self.pure_python_kernel(*(args + (output_grid,)))
            return output_grid

        if not self.specialized_sizes or\
                self.specialized_sizes != [y.shape for y in args]:
            self.specialized = SpecializedStencil(
                self.model, args, output_grid, self, self.testing
            )
            self.specialized_sizes = [arg.shape for arg in args]

        duration = c_float()
        # args = [arg.data for arg in args]
        args += (output_grid, byref(duration))
        self.specialized(*args)
        self.specialized.report(time=duration)
        # print("Took %.3fs" % duration.value)
        return output_grid

    def interior_points(self, x):
        dims = (range(self.ghost_depth[index], dim - self.ghost_depth[index])
                for index, dim in enumerate(x.shape))
        for item in itertools.product(*dims):
            yield tuple(item)

    def neighbors(self, point, neighbors_id=0):
        try:
            for neighbor in self.neighbor_definition[neighbors_id]:
                yield tuple(map(lambda a, b: a+b, list(point), list(neighbor)))
        except IndexError:
            # TODO: Make this a StencilException
            raise Exception("Undefined neighbor")

    def get_semantic_node(self, arg_names, *args):

        func_decl = PythonToStencilModel(arg_names).visit(
            get_ast(self.model)
        ).files[0].body[0]
        return StencilCall(func_decl, args[:-1], args[-1], self)

    def distance(self, x, y):
        """
        default euclidean distance override this to return something
        reasonable for each neighbor cell distance
        :param x: Point represented as a list or tuple
        :param y: Point represented as a list or tuple
        """
        return math.sqrt(sum([(x[i]-y[i])**2 for i in range(0, len(x))]))
