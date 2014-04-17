from stencil_code.stencil_kernel import StencilKernel
from stencil_code.stencil_grid import StencilGrid
import numpy as np
import unittest
import random
import math

height = 50
stdev_d = 3
stdev_s = 70
radius = 1
width = 2*8 + radius * 2


class TestOmpEndToEnd(unittest.TestCase):
    def setUp(self):
        out_grid1 = StencilGrid([width, width])
        out_grid1.ghost_depth = radius
        out_grid2 = StencilGrid([width, width])
        out_grid2.ghost_depth = radius
        in_grid = StencilGrid([width, width])
        in_grid.ghost_depth = radius

        for x in range(0, width):
            for y in range(0, width):
                in_grid.data[(x, y)] = random.random() * random.randint(0, 1000)

        for x in range(-radius, radius+1):
            for y in range(-radius, radius+1):
                in_grid.neighbor_definition[1].append((x, y))
        self.grids = (in_grid, out_grid1, out_grid2)

    def _check(self, test_kernel):
        in_grid, out_grid1, out_grid2 = self.grids
        test_kernel(backend='omp').kernel(in_grid, out_grid1)
        test_kernel(pure_python=True).kernel(in_grid, out_grid2)
        try:
            np.testing.assert_array_almost_equal(out_grid1.data, out_grid2.data)
        except:
            self.fail("Output grids not equal")

    def test_simple_ocl_kernel(self):
        class Kernel(StencilKernel):
            def kernel(self, in_grid, out_grid):
                for x in out_grid.interior_points():
                    for y in in_grid.neighbors(x, 1):
                        out_grid[x] += in_grid[y]
        self._check(Kernel)

    def test_2d_heat(self):
         class Kernel(StencilKernel):
            def kernel(self, in_img, out_img):
                for x in out_img.interior_points():
                    out_img[x] = in_img[x]
                    for y in in_img.neighbors(x, 0):
                        out_img[x] += 0.125 * in_img[y]
                    for z in in_img.neighbors(x, 1):
                        out_img[x] -= 0.125 * 2.0 * in_img[z]
         self._check(Kernel)

    def test_laplacian(self):
        alpha = 0.5
        beta = 1.0

        class LaplacianKernel(StencilKernel):
            def __init__(self, backend='c', pure_python=False):
                super(LaplacianKernel, self).__init__(backend=backend, pure_python=pure_python)
                self.constants = {'alpha': 0.5, 'beta': 1.0}

            def kernel(self, in_grid, out_grid):
                for x in in_grid.interior_points():
                    out_grid[x] = alpha * in_grid[x]
                    for y in in_grid.neighbors(x, 1):
                        out_grid[x] += beta * in_grid[y]
        self._check(LaplacianKernel)

    def test_bilateral_filter(self):
        width = 50
        height = 50
        stdev_d = 3
        stdev_s = 70
        # radius = stdev_d * 3
        radius = 3
        out_grid1 = StencilGrid([width, height])
        out_grid1.ghost_depth = radius
        out_grid2 = StencilGrid([width, height])
        out_grid2.ghost_depth = radius
        in_grid = StencilGrid([width, height])
        in_grid.ghost_depth = radius

        for x in range(0, width):
            for y in range(0, height):
                in_grid.data[(x, y)] = random.random() * random.randint(0, 255)

        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                in_grid.neighbor_definition[1].append((x, y))

        def gaussian(stdev, length):
            result = StencilGrid([length])
            scale = 1.0 / (stdev * math.sqrt(2.0 * math.pi))
            divisor = -1.0 / (2.0 * stdev * stdev)
            for x in range(length):
                result[x] = scale * math.exp(float(x) * float(x) * divisor)
            return result


        def distance(x, y):
            return math.sqrt(sum([(x[i] - y[i]) ** 2 for i in range(0, len(x))]))

        class Kernel(StencilKernel):
            def kernel(self, in_img, filter_d, filter_s, out_img):
                for x in out_img.interior_points():
                    for y in in_img.neighbors(x, 1):
                        out_img[x] += in_img[y] * filter_d[
                            int(distance(x, y))] * \
                                      filter_s[abs(int(in_img[x] - in_img[y]))]

        gaussian1 = gaussian(stdev_d, radius * 2)
        gaussian2 = gaussian(stdev_s, 256)

        Kernel(backend='omp').kernel(in_grid, gaussian1, gaussian2, out_grid1)
        Kernel(pure_python=True).kernel(in_grid, gaussian1, gaussian2, out_grid2)
        try:
            np.testing.assert_array_almost_equal(out_grid1.data, out_grid2.data)
        except:
            self.fail("Output grids not equal")

    def test_laplacian_kernel(self):
        alpha = 0.5
        beta = 1.0

        class LaplacianKernel(StencilKernel):
            def __init__(self, alpha, beta, pure_python=False):
                super(LaplacianKernel, self).__init__(pure_python=pure_python)
                self.constants = {'alpha': alpha, 'beta': beta}

            def kernel(self, in_grid, out_grid):
                for x in in_grid.interior_points():
                    out_grid[x] = alpha * in_grid[x]
                    for y in in_grid.neighbors(x, 1):
                        out_grid[x] += beta * in_grid[y]

        nx = 50
        ny = 50
        nz = 50
        input_grid = StencilGrid([nx, ny, nz])
        output_grid1 = StencilGrid([nx, ny, nz])
        output_grid2 = StencilGrid([nx, ny, nz])

        for x in input_grid.interior_points():
            input_grid[x] = random.randint(0, nx * ny * nz)

        laplacian = LaplacianKernel(alpha, beta)
        laplacian.kernel(input_grid, output_grid1)
        LaplacianKernel(alpha, beta, pure_python=True).kernel(input_grid,
                                                              output_grid2)
        try:
            np.testing.assert_array_almost_equal(output_grid1.data,
                                                 output_grid2.data)
        except:
            self.fail("Output grids not equal")