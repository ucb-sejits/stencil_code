from operator import mul
import numpy
from stencil_code.stencil_exception import StencilException
from stencil_code.backend.ocl_tools import product, OclTools

__author__ = 'chick'

from ctree.c.nodes import Lt, Constant, And, SymbolRef, Assign, Add, Mul, \
    Div, Mod, For, AddAssign, ArrayRef, FunctionCall, ArrayDef, Ref, \
    FunctionDecl, GtE, Sub, Cast
from hindemith.fusion.core import KernelCall


class BoundaryCopyKernel(object):
    """
    container for numbers necessary to generate and to call a kernel
    that copies boundary points from input to output during stencil
    copy boundary handling option
    """
    def __init__(self, halo, grid, dimension, device=None):
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
        self.dimensions = len(grid.shape)

        self.device = device

        self.global_size, self.global_offset = self.compute_global_size()
        self.local_size = OclTools.compute_local_size_1d()

        self.kernel_name = "boundary_copy_kernel_{}d_dim_{}".format(len(halo), dimension)

    def compute_global_size(self):
        dimension_sizes = [x for x in self.shape]
        dimension_offsets = [0 for _ in self.shape]
        for other_dimension in range(self.dimension):
            dimension_sizes[other_dimension] -= (2 * self.halo[other_dimension])
            dimension_offsets.append(self.halo[other_dimension])
        dimension_sizes[other_dimension] = self.halo[other_dimension]
        return dimension_sizes, dimension_offsets

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

