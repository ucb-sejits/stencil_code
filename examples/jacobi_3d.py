from __future__ import print_function
from ctree.util import Timer
from stencil_code.neighborhood import Neighborhood
from stencil_code.stencil_kernel import Stencil
import numpy as np

__author__ = 'chick'


class Jacobi3D(Stencil):
    def __init__(self, alpha, beta, backend='c',  boundary_handling='clamp'):
        super(Jacobi3D, self).__init__(
            backend=backend,
            boundary_handling=boundary_handling,
            neighborhoods=[Neighborhood.von_neuman_neighborhood(radius=1, dim=3, include_origin=False)]
        )
        self.alpha = alpha
        self.beta = beta

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            out_grid[x] = self.alpha * in_grid[x]
            for y in self.neighbors(x, 0):
                out_grid[x] += self.beta * in_grid[y]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Run jacobi stencil")
    parser.add_argument('d1', action="store", type=int, default=10)
    parser.add_argument('d2', action="store", type=int, default=10)
    parser.add_argument('d3', action="store", type=int, default=10)
    parser.add_argument('-l', '--log', action="store_true", dest="log")
    parser.add_argument('-r', '--random', action="store_true")
    parser.add_argument('-be', '--backend', action="store", dest="backend", default="c")
    parser.add_argument('-bh', '--boundary_handling', action="store", dest="boundary_handling", default="clamp")
    parser.add_argument('-pr', '--print-rows', action="store", dest="print_rows", type=int, default=-1)
    parser.add_argument('-a', '--alpha', action="store", type=float, default=0.0)
    parser.add_argument('-b', '--beta', action="store", type=float, default=1.0/6.0)
    parser.add_argument('-pc', '--print-cols', action="store", dest="print_cols", type=int, default=-1)

    parse_result = parser.parse_args()

    if parse_result.log:
        import logging
        logging.basicConfig(level=20)

    d1 = parse_result.d1
    d2 = parse_result.d2
    d3 = parse_result.d3
    print_d1 = min(10, d1)
    print_d2 = min(10, d2)
    print_d3 = min(10, d3)

    if parse_result.random:
        in_img = np.random.random([d1, d2, d3]).astype(np.float32)
    else:
        in_img = np.ones([d1, d2, d3]).astype(np.float32)

    stencil = Jacobi3D(alpha=parse_result.alpha, beta=parse_result.beta,
                       backend=parse_result.backend, boundary_handling=parse_result.boundary_handling)

    with Timer() as t:
        out_img = stencil(in_img)

    for i in range(print_d1-1, -1, -1):
        # print("i  {}".format(i))
        for j in range(print_d2-1, -1, -1):
            print(" "*j, end="")
            for k in range(print_d3):
                print("{:4.1f}".format(out_img[(i, j, k)]), end=" ")
            print()
        print()
    print()
    print("Jacobi process {}x{}x{} matrix in {} seconds".format(d1, d2, d3, t.interval))
