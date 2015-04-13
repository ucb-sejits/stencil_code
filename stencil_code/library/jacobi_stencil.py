from __future__ import print_function
from stencil_code.stencil_kernel import Stencil
import numpy
import numpy.testing


class Jacobi(Stencil):
    neighborhoods = [[(0, -1), (0, 1), (-1, 0), (1, 0)]]

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            out_grid[x] = 0.0
            for y in self.neighbors(x, 0):
                out_grid[x] += .25 * in_grid[y]


if __name__ == '__main__':  # pragma no cover
    import argparse

    parser = argparse.ArgumentParser("Run jacobi stencil")
    parser.add_argument('-r', '--rows', action="store", dest="rows", type=int, default=10)
    parser.add_argument('-c', '--cols', action="store", dest="cols", type=int, default=10)
    parser.add_argument('-l', '--log', action="store_true", dest="log")
    parser.add_argument('-b', '--backend', action="store", dest="backend", default="c")
    parser.add_argument('-bh', '--boundary_handling', action="store", dest="boundary_handling", default="clamp")
    parser.add_argument('-pr', '--print-rows', action="store", dest="print_rows", type=int, default=-1)
    parser.add_argument('-pc', '--print-cols', action="store", dest="print_cols", type=int, default=-1)
    parser.add_argument('-i', '--iterations', action="store", type=int, default=-1)


    parse_result = parser.parse_args()

    if parse_result.log:
        import logging
        logging.basicConfig(level=20)

    rows = parse_result.rows
    cols = parse_result.cols
    iterations = parse_result.iterations
    backend = parse_result.backend
    boundary_handling = parse_result.boundary_handling
    print_rows = parse_result.print_rows if parse_result.print_rows >= 0 else min(10, rows)
    print_cols = parse_result.print_cols if parse_result.print_cols >= 0 else min(10, cols)

    in_img = numpy.ones([rows, cols]).astype(numpy.float32)

    stencil = Jacobi(backend=backend, boundary_handling=boundary_handling)

    for trial in range(iterations):
        out_img = stencil(in_img)
    print("Optimal Local Size {}".format(stencil.specializer._tuner.manager.get_best_configuration()))

    # for index1 in range(print_rows):
    #     for index2 in range(print_cols):
    #         print("{:6s}".format(str(out_img[(index1, index2)])), end="")
    #     print()
