from __future__ import print_function
from stencil_code.stencil_kernel import Stencil
import numpy
import numpy.testing
from stencil_code.neighborhood import Neighborhood


class JacobiNd(Stencil):
    def __init__(self, dimensions, radius, use_moore=False, backend='c', boundary_handling='clamp', **kwargs):
        if use_moore:
            neighborhoods = [Neighborhood.moore_neighborhood(radius=radius, dim=dimensions, include_origin=False)]
        else:
            neighborhoods = [Neighborhood.von_neuman_neighborhood(radius=radius, dim=dimensions, include_origin=False)]

        super(JacobiNd, self).__init__(
            backend=backend, neighborhoods=neighborhoods, boundary_handling=boundary_handling, **kwargs)

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
    parser.add_argument('-l', '--log', action="store_true", dest="log")
    parser.add_argument('-b', '--backend', action="store", dest="backend", default="c")
    parser.add_argument('-bh', '--boundary_handling', action="store", dest="boundary_handling", default="clamp")
    parser.add_argument('-rd', '--random-data', action="store_true", dest="random_data", default=False)
    parser.add_argument('-m', '--moore_neighborhood', action="store_true", dest="use_moore", default=False,
                        help="Use Moore instead of von Neuman neighborhood")
    parser.add_argument('-i', '--iterations', action="store", type=int, dest="iterations", default=1)
    parser.add_argument('-r', '--neighborhood_radius', action="store", type=int, dest="radius", default=1)
    parser.add_argument('-pr', '--print-rows', action="store", dest="print_rows", type=int, default=-1)
    parser.add_argument('-pc', '--print-cols', action="store", dest="print_cols", type=int, default=-1)
    parse_result = parser.parse_args()

    if parse_result.log:
        import logging
        logging.basicConfig(level=20)

    num_dimensions = len(parse_result.dimension_sizes)

    if parse_result.random_data:
        in_img = numpy.random.random(parse_result.dimension_sizes).astype(numpy.float32)
    else:
        in_img = numpy.ones(parse_result.dimension_sizes).astype(numpy.float32)

    stencil = JacobiNd(
        dimensions=len(parse_result.dimension_sizes),
        radius=parse_result.radius,
        use_moore=parse_result.use_moore,
        backend=parse_result.backend,
        boundary_handling=parse_result.boundary_handling)

    out_img = in_img
    for _ in range(parse_result.iterations):
        out_img = stencil(in_img)

    def recursive_print(dim, higher_dims=None):
        if dim == num_dimensions - 1:
            for i in range(min(10, parse_result.dimension_sizes[dim])):
                index = higher_dims + (i, ) if higher_dims else i
                print(" {:6s}".format(str(out_img[i])), end="")
            print()
        elif dim == num_dimensions - 2:
            for index1 in range(min(10, parse_result.dimension_sizes[dim])):
                if higher_dims:
                    print(" " * (( min(10, parse_result.dimension_sizes[dim]) - index1 ) * 2), end="")
                for index2 in range(min(10, parse_result.dimension_sizes[dim+1])):
                    index = higher_dims + (index1, index2) if higher_dims else (index1, index2)
                    print(" {:6s}".format(str(out_img[index])), end="")
                print()
            print()
        else:
            for index in range(min(10, parse_result.dimension_sizes[dim])):
                sub_higher_dims = higher_dims + (index,) if higher_dims else (index,)
                recursive_print(dim+1, sub_higher_dims)

    recursive_print(0)