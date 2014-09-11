from stencil_code.stencil_kernel import StencilKernel
import numpy as np
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
    neighbor_definition = [[
        (-1, 1),  (0, 1),  (1, 1),
        (-1, 0),  (0, 0),  (1, 0),
        (-1, -1), (-1, 0), (-1, 1)
    ]]

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            for y in self.neighbors(x, 0):
                out_grid[x] += in_grid[y]


class TwoDHeatKernel(StencilKernel):
    neighbor_definition = [[(0, 0)], [
        (-1, 1),  (0, 1),  (1, 1),
        (-1, 0),  (0, 0),  (1, 0),
        (-1, -1), (-1, 0), (-1, 1)
    ]]

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
    neighbor_definition = [[(0, 0)], [
        (-1, 1),  (0, 1),  (1, 1),
        (-1, 0),  (0, 0),  (1, 0),
        (-1, -1), (-1, 0), (-1, 1)
    ]]

    @property
    def constants(self):
        return {'alpha': 0.5, 'beta': 1.0}

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(in_grid):
            out_grid[x] = alpha * in_grid[x]
            for y in self.neighbors(x, 0):
                out_grid[x] += beta * in_grid[y]


class BilatKernel(StencilKernel):
    neighbor_definition = [[
        (-1, 1),  (0, 1),  (1, 1),
        (-1, 0),  (0, 0),  (1, 0),
        (-1, -1), (-1, 0), (-1, 1)
    ]]

    def kernel(self, in_img, filter_d, filter_s, out_img):
        for x in self.interior_points(out_img):
            for y in self.neighbors(x, 0):
                out_img[x] += in_img[y] * filter_d[
                    int(distance(x, y))] * \
                    filter_s[abs(int(in_img[x] - in_img[y]))]


class Laplacian3DKernel(StencilKernel):
    neighbor_definition = [[
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (0, -1, -1), (0, -1, 0), (0, -1, 1),
        (1, -1, -1), (1, -1, 0), (1, -1, 1),
        (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
        (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
        (1, 1, -1), (1, 1, 0), (1, 1, 1)
    ]]
    constants = {'alpha': 0.5, 'beta': 1.0}

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(in_grid):
            out_grid[x] = alpha * in_grid[x]
            for y in self.neighbors(x, 0):
                out_grid[x] += beta * in_grid[y]
