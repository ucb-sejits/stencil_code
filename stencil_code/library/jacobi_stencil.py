from __future__ import print_function
from stencil_code.stencil_kernel import Stencil
import numpy
import numpy.testing


class Jacobi(Stencil):
    # neighborhoods = [[(0, -1), (0, 1)], [(-1, 0), (1, 0)]]

    # def kernel(self, in_grid, out_grid):
    #     for x in self.interior_points(out_grid):
    #         for y in self.neighbors(x, 0):
    #             out_grid[x] += .1 * in_grid[y]
    #         for y in self.neighbors(x, 1):
    #             out_grid[x] += .3 * in_grid[y]
    #
    neighborhoods = [[(0, -1)], [(-1, 0)], [(0, 1)], [(1, 0)]]

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            out_grid[x] = 1
            for y in self.neighbors(x, 0):
                out_grid[x] += 2 * in_grid[y]
            for y in self.neighbors(x, 1):
                out_grid[x] += 4 * in_grid[y]
            for y in self.neighbors(x, 2):
                out_grid[x] += 8 * in_grid[y]
            for y in self.neighbors(x, 3):
                out_grid[x] += 16 * in_grid[y]


if __name__ == '__main__':  # pragma no cover
    import sys
    import logging
    logging.basicConfig(level=20)

    height = 23 if len(sys.argv) < 2 else int(sys.argv[1])
    width = 23 if len(sys.argv) < 3 else int(sys.argv[2])

    # in_img = numpy.random.random([height, width]).astype(numpy.float32) * 100
    in_img = numpy.ones([height, width]).astype(numpy.float32)

    jacobi_stencil = Jacobi(backend='ocl')
    py = Jacobi(backend='c')

    out_img = jacobi_stencil(in_img)
    for i, r in enumerate(out_img):
        if i > len(out_img)-22:
            print("grid {:3d}  ".format(i), end="")
            for j, c in enumerate(r):
                if j < 60:
                    print("{!s:3s}".format(int(c)), end="")
            print()

    check = py(in_img)
    numpy.testing.assert_array_almost_equal(out_img, check, decimal=3)
