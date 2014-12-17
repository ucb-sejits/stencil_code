from stencil_code.stencil_kernel import Stencil
import numpy
import numpy.testing


class Jacobi(Stencil):
    neighborhoods = [[(0, -1), (0, 1)], [(-1, 0), (1, 0)]]

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            for y in self.neighbors(x, 0):
                out_grid[x] += .1 * in_grid[y]
            for y in self.neighbors(x, 1):
                out_grid[x] += .3 * in_grid[y]


if __name__ == '__main__':  # pragma no cover
    import sys
    import logging
    logging.basicConfig(level=20)

    height = 22 if len(sys.argv) < 2 else int(sys.argv[1])
    width = 25 if len(sys.argv) < 3 else int(sys.argv[2])

    in_img = numpy.random.random([height, width]).astype(numpy.float32) * 100

    jacobi_stencil = Jacobi(backend='ocl')
    py = Jacobi(backend='python')

    out_img = jacobi_stencil(in_img)
    check = py(in_img)
    numpy.testing.assert_array_almost_equal(out_img, check, decimal=4)
