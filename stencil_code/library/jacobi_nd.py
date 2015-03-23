from __future__ import print_function
from stencil_code.stencil_kernel import Stencil
import numpy
import numpy.testing
from stencil_code.neighborhood import Neighborhood


class JacobiNd(Stencil):
    def __init__(self, backend, dimensions, boundary_handling, **kwargs):
        neighborhoods = [Neighborhood.von_neuman_neighborhood(radius=1, dim=dimensions, include_origin=False)]
        super(JacobiNd, self).__init__(backend, neighborhoods, boundary_handling, **kwargs)
        self.neighbor_weight = 1.0 / len(neighborhoods[0])

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            out_grid[x] = 0.0
            for y in self.neighbors(x, 0):
                out_grid[x] += self.neighbor_weight * in_grid[y]


if __name__ == '__main__':  # pragma no cover
    import argparse

    parser = argparse.ArgumentParser("Run jacobi stencil")
    parser.add_argument("dimension_sizes", metavar='N', type=int, nargs='+', help="size of each dimension")
    parser.add_argument('-d', '--dimensions', action="store", dest="dimensions", type=int, default=2)
    parser.add_argument('-l', '--log', action="store_true", dest="log")
    parser.add_argument('-b', '--backend', action="store", dest="backend", default="c")
    parser.add_argument('-bh', '--boundary_handling', action="store", dest="boundary_handling", default="clamp")
    parser.add_argument('-pr', '--print-rows', action="store", dest="print_rows", type=int, default=-1)
    parser.add_argument('-pc', '--print-cols', action="store", dest="print_cols", type=int, default=-1)
    parse_result = parser.parse_args()

    if parse_result.log:
        import logging
        logging.basicConfig(level=20)

    dimensions = parse_result.dimensions
    dimension_sizes = parse_result.dimension_sizes
    if dimensions != len(dimension_sizes):
        print("number of dimensions must agree with dimension sizes")
        exit(1)

    backend = parse_result.backend
    boundary_handling = parse_result.boundary_handling

    in_img = numpy.ones(dimension_sizes).astype(numpy.float32)
    # in_img[5:6, 5:9, 5:6] = 20

    stencil = JacobiNd(backend=backend, dimensions=dimensions, boundary_handling=boundary_handling)

    out_img = stencil(in_img)

    def recursive_print(dim, higher_dims=None):
        if dim == dimensions - 1:
            for i in range(max(10, dimension_sizes[dim])):
                index = higher_dims + (i, ) if higher_dims else i
                print("{:6s}".format(str(out_img[i])), end="")
            print()
        elif dim == dimensions - 2:
            for index1 in range(max(10, dimension_sizes[dim])):
                if higher_dims:
                    print(" " * (( max(10, dimension_sizes[dim]) - index1 ) * 2), end="")
                for index2 in range(max(10, dimension_sizes[dim+1])):
                    index = higher_dims + (index1, index2) if higher_dims else (index1, index2)
                    print("{:6s}".format(str(out_img[index])), end="")
                print()
        else:
            for index in range(max(10, dimension_sizes[dim])):
                sub_higher_dims = higher_dims + (index,) if higher_dims else (index,)
                recursive_print(dim+1, sub_higher_dims)

    recursive_print(0)