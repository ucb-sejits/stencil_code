from stencil_code.stencil_kernel import StencilKernel
from stencil_code.stencil_grid import StencilGrid
import numpy as np
import unittest
import math

stdev_d = 3
stdev_s = 70
radius = 1
width = 64 + radius * 2
height = width


def gaussian(stdev, length):
    result = np.empty([length])
    scale = 1.0 / (stdev * math.sqrt(2.0 * math.pi))
    divisor = -1.0 / (2.0 * stdev * stdev)
    for x in range(length):
        result[x] = scale * math.exp(float(x) * float(x) * divisor)
    return result

gaussian1 = gaussian(stdev_d, radius * 2)
gaussian2 = gaussian(stdev_s, 256)


def distance(x, y):
    return math.sqrt(
        sum([(x[i] - y[i]) ** 2 for i in range(0, len(x))]))


class SimpleKernel(StencilKernel):
    @property
    def dim(self):
        return 2

    @property
    def ghost_depth(self):
        return 1

    def neighbors(self, pt, defn=0):
        if defn == 0:
            for x in range(-radius, radius+1):
                for y in range(-radius, radius+1):
                    yield (pt[0] - x, pt[1] - y)

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            for y in self.neighbors(x, 0):
                out_grid[x] += in_grid[y]


class TwoDHeatKernel(StencilKernel):
    @property
    def dim(self):
        return 2

    @property
    def ghost_depth(self):
        return 1

    def neighbors(self, pt, defn=0):
        if defn == 0:
            yield pt
        elif defn == 1:
            yield pt
            yield pt[0] - 1, pt[1]
            yield pt[0] - 1, pt[1] - 1
            yield pt[0] - 1, pt[1] + 1
            yield pt[0], pt[1]
            yield pt[0], pt[1] - 1
            yield pt[0], pt[1] + 1
            yield pt[0] + 1, pt[1]
            yield pt[0] + 1, pt[1] - 1
            yield pt[0] + 1, pt[1] + 1

    def kernel(self, in_img, out_img):
        for x in self.interior_points(out_img):
            out_img[x] = in_img[x]
            for y in self.neighbors(x, 0):
                out_img[x] += 0.125 * in_img[y]
            for z in self.neighbors(x, 1):
                out_img[x] -= 0.125 * 2.0 * in_img[z]


alpha = 0.5
beta = 1.0


class LaplacianKernel(StencilKernel):
    @property
    def dim(self):
        return 2

    @property
    def ghost_depth(self):
        return 1

    @property
    def constants(self):
        return {'alpha': 0.5, 'beta': 1.0}

    def neighbors(self, pt, defn=0):
        if defn == 1:
            for x in range(-radius, radius+1):
                for y in range(-radius, radius+1):
                    yield (pt[0] - x, pt[1] - y)

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(in_grid):
            out_grid[x] = alpha * in_grid[x]
            for y in self.neighbors(x, 1):
                out_grid[x] += beta * in_grid[y]

class BilatKernel(StencilKernel):
    @property
    def dim(self):
        return 2

    @property
    def ghost_depth(self):
        return radius

    def neighbors(self, pt, defn=0):
        if defn == 1:
            for x in range(-radius, radius+1):
                for y in range(-radius, radius+1):
                    yield (pt[0] - x, pt[1] - y)

    def kernel(self, in_img, filter_d, filter_s, out_img):
        for x in self.interior_points(out_img):
            for y in self.neighbors(x, 1):
                out_img[x] += in_img[y] * filter_d[
                    int(distance(x, y))] * \
                    filter_s[abs(int(in_img[x] - in_img[y]))]


class TestOclEndToEnd(unittest.TestCase):
    def setUp(self):
        # out_grid1 = StencilGrid([width, width])
        # out_grid1.ghost_depth = radius
        # out_grid2 = StencilGrid([width, width])
        # out_grid2.ghost_depth = radius
        in_grid = np.random.rand(width, width).astype(np.float32) * 1000
        out_grid1 = np.zeros_like(in_grid)
        out_grid2 = np.zeros_like(in_grid)
        # data = random.rand(width, width).astype(np.float32) * 1000
        # in_grid = StencilGrid([width, width], data=data)
        # in_grid.ghost_depth = radius

        # for x in range(-radius, radius+1):
        #     for y in range(-radius, radius+1):
        #         in_grid.neighbor_definition[1].append((x, y))
        self.grids = (in_grid, out_grid1, out_grid2)

    def _check(self, test_kernel):
        in_grid, out_grid1, out_grid2 = self.grids
        out_grid1 = test_kernel(backend='ocl', testing=True).kernel(in_grid)
        out_grid2 = test_kernel(pure_python=True).kernel(in_grid)
        try:
            np.testing.assert_array_almost_equal(out_grid1, out_grid2,
                                                 decimal=3)
        except AssertionError as e:
            self.fail("Output grids not equal: %s" % e.message)

    def test_simple_ocl_kernel(self):
        self._check(SimpleKernel)

    def test_2d_heat(self):
        self._check(TwoDHeatKernel)

    def test_laplacian(self):
        self._check(LaplacianKernel)

    def test_bilateral_filter(self):
        in_grid = np.random.rand(width, height).astype(np.float32) * 255

        out_grid1 = BilatKernel(backend='ocl', testing=True).kernel(
            in_grid, gaussian1, gaussian2)
        out_grid2 = BilatKernel(pure_python=True).kernel(
            in_grid, gaussian1, gaussian2)
        try:
            np.testing.assert_array_almost_equal(out_grid1, out_grid2)
        except AssertionError as e:
            self.fail("Output grids not equal: %s" % e.message)
