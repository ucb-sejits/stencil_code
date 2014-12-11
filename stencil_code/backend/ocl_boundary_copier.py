from __future__ import print_function
import numpy
import numpy.random
from stencil_code.stencil_exception import StencilException
from stencil_code.backend.ocl_tools import OclTools

__author__ = 'chick'

import ctypes as ct
from ctree.c.nodes import Lt, Constant, And, SymbolRef, Assign, Add, Mul, \
    AddAssign, ArrayRef, FunctionCall, If
from ctree.ocl.macros import get_global_id


class OclBoundaryCopier(object):
    """
    container for numbers necessary to generate and to call a kernel
    that copies boundary points from input to output during stencil
    copy boundary handling option
    """
    def __init__(self, halo, grid, dimension, in_grid_name="in_grid", out_grid_name="out_grid", device=None):
        """

        :param halo: the halo shape
        :param grid: the shape of the grid that stencil is to be applied to
        :param dimension: the dimension this kernel applies to
        :return:
        """
        self.halo = tuple(halo)
        self.grid = grid
        self.shape = grid.shape
        self.in_grid_name = in_grid_name
        self.out_grid_name = out_grid_name

        # check for some pathologies and raise exception if any are present
        if len(halo) != len(self.shape):
            raise StencilException("halo {} can't apply to grid shape {}".format(self.halo, self.shape))

        if dimension < 0 or dimension >= len(self.shape):
            raise StencilException("dimension {} too big for grid shape {}".format(dimension, self.shape))

        # halo or grid to small
        if any([x < 1 or y < 1 for x, y in zip(self.halo, self.shape)]):
            raise StencilException(
                "halo {} can't be bigger than grid {} in any dimension".format(self.halo, self.shape)
            )
        # no interior points in a dimension
        if any([s <= 2*h for h, s in zip(self.halo, self.shape)]):
            raise StencilException(
                "halo {} can't span grid shape {} in any dimension".format(self.halo, self.shape)
            )

        self.dimension = dimension
        self.dimensions = len(grid.shape)

        self.device = device

        self.global_size, self.global_offset = self.compute_global_size()
        self.local_size = OclTools(device=None).compute_local_size(self.global_size)
        self.virtual_global_size = self.compute_virtual_global_size()

        self.kernel_name = OclBoundaryCopier.kernel_name(self.dimension)

    @staticmethod
    def kernel_name(dimension):
        return "kernel_d{}".format(dimension)

    def compute_global_size(self):
        """
        a boundary kernel will for the edge points in one dimension iterated over
        all the points in higher indexed dimensions and interior points in
        lower indexed dimensions.  Because global_size for this kernel is a subset of the
        orginal grid it must be offset from the origin. OpenCL does not currently support this so
        we must implement it ourselves in the kernel code
        :return:
        """
        dimension_sizes = [x for x in self.shape]
        dimension_offsets = [0 for _ in self.shape]
        for other_dimension in range(self.dimension):
            dimension_sizes[other_dimension] -= max(1, (2 * self.halo[other_dimension]))
            dimension_offsets[other_dimension] = self.halo[other_dimension]
        dimension_sizes[self.dimension] = self.halo[self.dimension]
        return dimension_sizes, dimension_offsets

    def compute_virtual_global_size(self):
        """
        if the global size is not a even multiple of the local size then the virtual size will be
        the next largest multiple
        :return:
        """
        return [
            size if size % self.local_size[dim] == 0 else (int((size / self.local_size[dim]) + 1) * self.local_size[dim])
            for dim, size in enumerate(self.global_size)
        ]

    def generate_ocl_kernel_body(self):
        """
        generate OpenCL code to handle this slice of the boundary,
        kernel will have threads complete the low order indices first
        then will do the high order indices, i.e. the points in the ghost zone of the given dimension
        :return:
        """

        # copy boundary points from in_grid to out_grid

        body = [
            Assign(SymbolRef('global_index', ct.c_int()), self.gen_global_index()),

            self.gen_index_in_bounds_conditional(
                Assign(
                    ArrayRef(SymbolRef(self.out_grid_name), SymbolRef('global_index')),
                    ArrayRef(SymbolRef(self.in_grid_name), SymbolRef('global_index'))
                ),
                is_low_side=True
            ),

            FunctionCall(SymbolRef("barrier"), [SymbolRef("CLK_LOCAL_MEM_FENCE")]),

            Assign(
                SymbolRef("global_index"),
                self.gen_global_index_with_halo_offset()
            ),

            self.gen_index_in_bounds_conditional(
                Assign(
                    ArrayRef(SymbolRef(self.out_grid_name), SymbolRef('global_index')),
                    ArrayRef(SymbolRef(self.in_grid_name), SymbolRef('global_index'))
                ),
                is_low_side=False
            )
        ]

        return body

    def gen_global_index(self):
        dim = self.dimensions
        index = get_global_id(dim - 1)
        for d in reversed(range(dim - 1)):
            stride = self.grid.strides[d] // \
                self.grid.itemsize
            index = Add(
                index,
                Mul(
                    Add(
                        get_global_id(d),
                        Constant(self.global_offset[d])
                    ),
                    Constant(stride)
                )
            )
        return index

    def gen_global_index_with_halo_offset(self):
        dim = self.dimensions
        index = get_global_id(dim - 1)
        if dim - 1 == self.dimension:
            index = Add(index, Constant(self.shape[dim-1] - self.halo[dim-1]))
        for d in reversed(range(dim - 1)):
            stride = self.grid.strides[d] // \
                self.grid.itemsize
            add_amount = Add(get_global_id(d), Constant(self.global_offset[d]))
            if d == self.dimension:
                add_amount = Add(add_amount, Constant(self.shape[self.dimension] - self.halo[self.dimension]))
            index = Add(
                index,
                Mul(
                    add_amount,
                    Constant(stride)
                )
            )
        return index

    def gen_index_in_bounds_conditional(self, body, is_low_side=True):
        """
        provide bounds checking if the virtual grid size differs from grid size
        :param body:
        :return:
        """
        def is_conditional_required_for(d):
            if self.virtual_global_size[d] == self.global_size[d]:
                return False
            if d == self.dimension and is_low_side:
                return False
            return True

        conditional = None
        for dim in range(self.dimensions):
            if is_conditional_required_for(dim):
                if conditional is None:
                    conditional = Lt(get_global_id(dim), Constant(self.global_size[dim]))
                else:
                    conditional = And(conditional, Lt(get_global_id(dim), Constant(self.global_size[dim])))

        if conditional is None:
            return body
        else:
            return If(conditional, body)


def boundary_kernel_factory(halo, grid, in_grid_name="in_grid", out_grid_name="out_grid", device=None):
    return [
        OclBoundaryCopier(halo, grid, dimension, in_grid_name=in_grid_name, out_grid_name=out_grid_name, device=device)
        for dimension in range(len(grid.shape))
    ]


if __name__ == '__main__':  # pragma no cover
    import itertools
    import random

    # grid = numpy.ones([11, 513])
    # halo = [1, 2]
    # for dim in range(len(halo)):
    #     bk = BoundaryCopyKernel(halo, grid, dimension=dim, device=None)
    #
    #     print(
    #         "gs {} ls {} code {}".format(
    #             bk.global_size, bk.local_size, [x.codegen() for x in bk.generate_ocl_kernel()]
    #         )
    #     )
    #     # ctree.browser_show_ast(bk.generate_ocl_kernel()[1])
    #     # print(bk.generate_ocl_kernel())
    # exit(0)
    #
    # grid = numpy.ones([4, 8, 32])
    # halo = [1, 2, 4]
    # for dim in range(len(halo)):
    #     bk = BoundaryCopyKernel(halo, grid, dimension=dim, device=None)
    #
    #     print([x.codegen() for x in bk.generate_ocl_kernel()])
    #     ctree.browser_show_ast(bk.generate_ocl_kernel()[1])
    #     # print(bk.generate_ocl_kernel())
    # exit(0)
    #
    # numpy.random.seed(0)

    for dims in range(1, 4):
        shape_list = [8, 1014, 4096][:dims]
        for _ in range(dims):
            shape = numpy.random.random(shape_list).astype(numpy.int) + 1

            print("Dimension {} {}".format(dims, "="*80))
            for shape in itertools.permutations(shape_list):
                rand_shape = map(lambda x: random.randint(5, x), shape)
                in_grid = numpy.zeros(rand_shape)
                ghost_depth = map(lambda x: random.randint(1, (max(2, (x-1)/2))), rand_shape)

                print("shape {} halo {}".format(rand_shape, ghost_depth))
                boundary_kernels = boundary_kernel_factory(ghost_depth, in_grid)

                for boundary_dimension, bk0 in enumerate(boundary_kernels):
                    print("dim {} {:16} {:16} ".format(
                        boundary_dimension, in_grid.shape, ghost_depth
                    ), end="")
                    print("global {:16} local {:16} {:16}".format(
                        bk0.global_size, bk0.local_size, bk0.global_offset
                    ))
