from operator import mul
import random
import numpy
from stencil_code.stencil_exception import StencilException

__author__ = 'chick'

from ctree.c.nodes import Lt, Constant, And, SymbolRef, Assign, Add, Mul, \
    Div, Mod, For, AddAssign, ArrayRef, FunctionCall, ArrayDef, Ref, \
    FunctionDecl, GtE, Sub, Cast
from hindemith.fusion.core import KernelCall


def product(vector):
    reduce(mul, vector, 1)


class BoundaryCopyKernel(object):
    """
    container for numbers necessary to generate and to call a kernel
    that copies boundary points from input to output during stencil
    copy boundary handling option
    """
    def __init__(self, halo, grid, dimension, max_work_group_size=128):
        """

        :param halo: the halo shape
        :param grid: the shape of the grid that stencil is to be applied to
        :param dimension: the dimension this kernel applies to
        :return:
        """
        self.halo = tuple(halo)
        self.grid = grid
        self.shape = grid.shape
        self.dimension = dimension

        self.max_work_group_size = max_work_group_size
        self.max_compute_units = 40  # Todo: get this from outside
        self.max_local_group_sizes = [512, 512, 512]

        self.grid_size = reduce(mul, self.shape)
        self.local_size = self.compute_local_group_size()
        self.global_size, self.global_offset = self.compute_global_size()
        self.local_size = self.compute_local_group_size()
        self.padding = [0 for _ in range(len(shape))]

        self.kernel_name = "boundary_copy_kernel_{}d_dim_{}".format(len(halo), dimension)

    def compute_global_size(self):
        global_size = [0 for _ in self.grid.shape]
        global_offset = [0 for _ in self.grid.shape]

        for other_dim in range(len(self.grid.shape)):
            if other_dim < self.dimension:
                global_size[other_dim] = self.grid.shape[other_dim] - 2 * self.halo[other_dim]
                global_offset[other_dim] = self.halo[other_dim]
            elif other_dim == self.dimension:
                global_size[other_dim] = self.halo[other_dim]
                global_offset[other_dim] = 0
            else:
                global_size[other_dim] = self.grid.shape[other_dim]
                global_offset[other_dim] = 0

        return global_size, global_offset

    def compute_local_size_1d(self):
        return (min(self.grid_size, self.max_local_group_sizes[0], self.shape[0]/2), )

    def compute_local_size_2d(self):
        d0_size, d1_size = 1, 1

        if self.grid_size < self.max_work_group_size:
            return self.shape

        while True:
            if self.shape[0] % 2 == 1:
                d0_size = 1
            else:
                d0_size = min(self.max_local_group_sizes[0], d0_size * 2)
            if self.grid_size <= d0_size * d1_size:
                return d0_size, d1_size

            if self.shape[1] % 2 == 1:
                d1_size = 1
            else:
                d1_size = min(self.max_local_group_sizes[1], d1_size * 2)

            if self.grid_size - d0_size * d1_size <= 0:
                return d0_size, d1_size

            if d0_size == self.shape[0] or d1_size == self.shape[1]:
                return d0_size, d1_size

    def compute_local_size_3d(self):
        d0_size, d1_size, d2_size = 1, 1, 1

        if self.grid_size < self.max_work_group_size:
            return self.shape

        while True:
            if self.shape[0] % 2 == 1:
                d0_size = 1
            else:
                d0_size = min(self.max_local_group_sizes[0], d0_size * 2)
            if self.grid_size <= d0_size * d1_size * d2_size:
                return d0_size, d1_size, d2_size

            if self.shape[1] % 2 == 1:
                d1_size = 1
            else:
                d1_size = min(self.max_local_group_sizes[1], d1_size * 2)
            if self.grid_size <= d0_size * d1_size * d2_size:
                return d0_size, d1_size, d2_size

            if self.shape[2] % 2 == 1:
                d2_size = 1
            else:
                d2_size = min(self.max_local_group_sizes[2], d2_size * 2)
            if self.grid_size <= d0_size * d1_size * d2_size:
                return d0_size, d1_size, d2_size

            if d0_size == self.shape[0] or d1_size == self.shape[1] or d2_size == self.shape[2]:
                return d0_size, d1_size, d2_size

    def compute_local_group_size(self):
        if len(self.shape) == 1:
            return self.compute_local_size_1d()
        elif len(self.shape) == 2:
            return self.compute_local_size_2d()
        elif len(self.shape) == 3:
            return self.compute_local_size_3d()
        else:
            raise StencilException("ocl stencils must be 1, 2 or 3 dimensions")

    def make_kernel_template(self):
        """
        a boundary kernel will be created, one for each dimension. That kernel
        will iterate over all other indices of the other dimensions, bounded by those points
        that have been handled by a previous kernel.
        The global_size will the entirety of the end planes in the minor side of the dimension
        :return:
        """

        boundary_kernel = FunctionDecl()
        kernel_call = KernelCall()
        template_string = """
            __kernel void {kernel_name}(__global const float* in_grid, __global float* out_grid, __local float* block) {
                #define global_array_macro(d0, d1) ((d1)+((d0) * {actual_width}))
                int minor_global_index = get_global_id(1) + get_global_id(0) * {actual_width};
                int major_global_index = (get_global_id(1) + {interior_width} + get_global_id(0) * {actual_width};

                out_grid[minor_global_index] = in_grid[minor_global_index];
                memfence(CLK_GLOBAL_MEM_FENCE);
                out_grid[major_global_index] = in_grid[major_global_index];
            };

            )))
        """

if __name__ == '__main__':
    import itertools

    numpy.random.seed(0)

    for dims in range(1, 4):
        shape_list = [8, 1014, 4096][:dims]
        for _ in range(dims):
            shape = numpy.random.random(shape_list).astype(numpy.int) + 1
            halo = [2, 3, 5][:dims]

            print("Dimension {} {}".format(dims, "="*80))
            for shape in itertools.permutations(shape_list):
                rand_shape = map(lambda x: numpy.random.randint(1, x), shape)
                print("rand_shape {}".format(rand_shape))
                halo = [2, 3, 5][:dims]
                in_grid = numpy.zeros(rand_shape)
                bk0 = BoundaryCopyKernel(halo, in_grid, 0)
                print("{:16s} {:16s} local {:16s}".format(bk0.shape, bk0.halo, bk0.local_size))

