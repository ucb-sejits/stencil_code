from stencil_code.stencil_kernel import StencilKernel
import numpy
import logging

logging.basicConfig(level=20)


class Jacobi(StencilKernel):
    neighbor_definition = [[(0, -1), (0, 1)], [(-1, 0), (1, 0)]]

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            for y in self.neighbors(x, 0):
                out_grid[x] += .1 *in_grid[y]
            for y in self.neighbors(x, 1):
                out_grid[x] += .3 * in_grid[y]

in_img = numpy.random.rand(1024, 1024).astype(numpy.float32) * 100

jacobi_stencil = Jacobi(backend='ocl')
py = Jacobi(backend='python')

out_img = jacobi_stencil(in_img)
check = py(in_img)
numpy.testing.assert_array_almost_equal(out_img[1:-1, 1:-1], check[1:-1, 1:-1])
