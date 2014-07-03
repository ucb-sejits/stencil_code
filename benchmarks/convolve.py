from scipy.ndimage import convolve
import numpy as np
from stencil_code.stencil_grid import StencilGrid
from stencil_code.stencil_kernel import StencilKernel

import time


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
x = []
iterations = 10
total1 = 0.0
results1 = []
for _ in range(iterations):
    with Timer() as t:
        out_image = convolve(image, stencil, mode='constant', cval=0.0)
    total1 += t.interval
    results1.append(t.interval)
    x.append(width)
print("Numpy convolve avg: {0}".format(total1/iterations))

out_grid = StencilGrid([width, height])

out_grid.ghost_depth = 2
total2 = 0.0

results2 = []
for _ in range(iterations):
    with Timer() as t:
        Kernel(backend='c').kernel(in_grid, out_grid)
    total2 += t.interval
    results2.append(t.interval)
print("Specialized C with compile time avg: {0}".format(total2/iterations))

np.testing.assert_array_almost_equal(out_grid[2:-2, 2:-2], out_image[2:-2, 2:-2], decimal=3)

kernel = Kernel(backend='c')
kernel.kernel(in_grid, out_grid)
total3 = 0.0
results3 = []
for _ in range(iterations):
    with Timer() as t:
        kernel.kernel(in_grid, out_grid)
    total3 += t.interval
    results3.append(t.interval)
print("Specialized C time avg without compile {0}".format(total3/iterations))

results4 = []
total4 = 0.0
for _ in range(iterations):
    with Timer() as t:
        Kernel(backend='omp').kernel(in_grid, out_grid)
    total4 += t.interval
    results4.append(t.interval)
print("Specialized OpenMP with compile time avg: {0}".format(total4/iterations))

np.testing.assert_array_almost_equal(out_grid[2:-2, 2:-2], out_image[2:-2, 2:-2], decimal=3)

kernel = Kernel(backend='omp')
kernel.kernel(in_grid, out_grid)
total5 = 0.0
results5 = []
for _ in range(iterations):
    with Timer() as t:
        kernel.kernel(in_grid, out_grid)
    total5 += t.interval
    results5.append(t.interval)
print("Specialized OpenMP time avg without compile {0}".format(total5/iterations))

results6 = []
total6 = 0.0
for _ in range(iterations):
    with Timer() as t:
        Kernel(backend='ocl').kernel(in_grid, out_grid)
    total6 += t.interval
    results6.append(t.interval)
print("Specialized OpenCL with compile time avg: {0}".format(total6/iterations))

np.testing.assert_array_almost_equal(out_grid[2:-2, 2:-2], out_image[2:-2, 2:-2], decimal=3)

kernel = Kernel(backend='ocl')
kernel.kernel(in_grid, out_grid)
total7 = 0.0
results7 = []
for _ in range(iterations):
    with Timer() as t:
        kernel.kernel(in_grid, out_grid)
    total7 += t.interval
    results7.append(t.interval)
print("Specialized OpenCL time avg without compile {0}".format(total7/iterations))

colors = ['b', 'c', 'y', 'm', 'r']
import matplotlib.pyplot as plt

r1 = plt.scatter(x, results1, marker='x', color=colors[0])
r2 = plt.scatter(x, results2, marker='x', color=colors[1])
r3 = plt.scatter(x, results3, marker='x', color=colors[2])
r4 = plt.scatter(x, results4, marker='x', color=colors[3])
r5 = plt.scatter(x, results5, marker='x', color=colors[4])
r6 = plt.scatter(x, results6, marker='o', color=colors[0])
r7 = plt.scatter(x, results7, marker='o', color=colors[1])

plt.legend((r1, r2, r3, r4, r5, r6, r7),
           ('Numpy convolve', 'C with compile', 'C without compile', 'OpenMP with compile', 'OpenMp withouth compile', 'OpenCL with compile', 'OpenCL without compile'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
plt.show()

# print("Print numpy out_image: ", out_image)
# print("Print numpy out_image2: ", out_image2)
# print("StencilGrid out_grid:", out_grid)
