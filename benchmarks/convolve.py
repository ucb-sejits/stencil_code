from scipy.ndimage import convolve
import numpy as np
from stencil_code.stencil_grid import StencilGrid
from stencil_code.stencil_kernel import StencilKernel

import time
import logging

# logging.basicConfig(level=20)


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.interval = time.clock() - self.start

width = 2**10 + 4
height = width

image = np.random.rand(width, height)
# print("Print numpy image: ", image)
stencil = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, -4, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
)


class Kernel(StencilKernel):
    def kernel(self, in_grid, out_grid):
        for x in in_grid.interior_points():
            out_grid[x] = -4 * in_grid[x]
            for y in in_grid.neighbors(x, 1):
                out_grid[x] += in_grid[y]

in_grid = StencilGrid([width, height])
for i in range(width):
    for j in range(height):
        in_grid[(i, j)] = image[(i, j)]

in_grid.neighbor_definition[1] = [
    (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2),
    (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1),
    (-2, 0), (-1, 0), (1, 0), (2, 0),
    (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
    (-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2)
]
in_grid.ghost_depth = 2

# print("StencilGrid in_grid:", in_grid)
iterations = 10
total = 0.0
for _ in range(iterations):
    with Timer() as t:
        out_image = convolve(image, stencil, mode='constant', cval=0.0)
    total += t.interval
print("Numpy convolve avg: {0}".format(total/iterations))

out_grid = StencilGrid([width, height])

out_grid.ghost_depth = 2
total = 0.0
for _ in range(iterations):
    with Timer() as t:
        Kernel(backend='c').kernel(in_grid, out_grid)
    total += t.interval
print("Specialized C with compile time avg: {0}".format(total/iterations))

np.testing.assert_array_almost_equal(out_grid[2:-2, 2:-2], out_image[2:-2, 2:-2], decimal=3)

kernel = Kernel(backend='c')
kernel.kernel(in_grid, out_grid)
total = 0.0
for _ in range(iterations):
    with Timer() as t:
        kernel.kernel(in_grid, out_grid)
    total += t.interval
print("Specialized C time avg without compile {0}".format(total/iterations))

for _ in range(iterations):
    with Timer() as t:
        Kernel(backend='omp').kernel(in_grid, out_grid)
    total += t.interval
print("Specialized OpenMP with compile time avg: {0}".format(total/iterations))

np.testing.assert_array_almost_equal(out_grid[2:-2, 2:-2], out_image[2:-2, 2:-2], decimal=3)

kernel = Kernel(backend='omp')
kernel.kernel(in_grid, out_grid)
total = 0.0
for _ in range(iterations):
    with Timer() as t:
        kernel.kernel(in_grid, out_grid)
    total += t.interval
print("Specialized OpenMP time avg without compile {0}".format(total/iterations))

for _ in range(iterations):
    with Timer() as t:
        Kernel(backend='ocl').kernel(in_grid, out_grid)
    total += t.interval
print("Specialized OpenCL with compile time avg: {0}".format(total/iterations))

np.testing.assert_array_almost_equal(out_grid[2:-2, 2:-2], out_image[2:-2, 2:-2], decimal=3)

kernel = Kernel(backend='ocl')
kernel.kernel(in_grid, out_grid)
total = 0.0
for _ in range(iterations):
    with Timer() as t:
        kernel.kernel(in_grid, out_grid)
    total += t.interval
print("Specialized OpenCL time avg without compile {0}".format(total/iterations))

# print("Print numpy out_image: ", out_image)
# print("Print numpy out_image2: ", out_image2)
# print("StencilGrid out_grid:", out_grid)
