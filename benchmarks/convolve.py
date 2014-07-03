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

x = []
iterations = 10
results = [[] for _ in range(7)]
totals = [0.0 for _ in range(7)]

for width in (2**x + 4 for x in range(10, 13)):
    height = width

    image = np.random.rand(width, height)

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
    out_grid = StencilGrid([width, height])
    out_grid.ghost_depth = 2

    # print("StencilGrid in_grid:", in_grid)

    c_kernel = Kernel(backend='c')
    c_kernel.kernel(in_grid, out_grid)

    omp_kernel = Kernel(backend='omp')
    omp_kernel.kernel(in_grid, out_grid)

    ocl_kernel = Kernel(backend='ocl')
    ocl_kernel.kernel(in_grid, out_grid)

    for _ in range(iterations):
        with Timer() as t0:
            out_image = convolve(image, stencil, mode='constant', cval=0.0)
        totals[0] += t0.interval
        results[0].append(t0.interval)
        x.append(width)

        with Timer() as t1:
            Kernel(backend='c').kernel(in_grid, out_grid)
        totals[1] += t1.interval
        results[1].append(t1.interval)

        with Timer() as t2:
            c_kernel.kernel(in_grid, out_grid)
        totals[2] += t2.interval
        results[2].append(t2.interval)

        with Timer() as t3:
            Kernel(backend='omp').kernel(in_grid, out_grid)
        totals[3] += t3.interval
        results[3].append(t3.interval)

        with Timer() as t4:
            omp_kernel.kernel(in_grid, out_grid)
        totals[4] += t4.interval
        results[4].append(t4.interval)

        with Timer() as t5:
            Kernel(backend='ocl').kernel(in_grid, out_grid)
        totals[5] += t5.interval
        results[5].append(t5.interval)

        with Timer() as t6:
            ocl_kernel.kernel(in_grid, out_grid)
        totals[6] += t6.interval
        results[6].append(t6.interval)

    print("---------- Results for dim {0}x{1} ----------".format(width, height))
    print("Numpy convolve avg: {0}".format(totals[0]/iterations))
    print("Specialized C with compile time avg: {0}".format(totals[1]/iterations))
    print("Specialized C time avg without compile {0}".format(totals[2]/iterations))
    print("Specialized OpenMP with compile time avg: {0}".format(totals[3]/iterations))
    print("Specialized OpenMP time avg without compile {0}".format(totals[4]/iterations))
    print("Specialized OpenCL with compile time avg: {0}".format(totals[5]/iterations))
    print("Specialized OpenCL time avg without compile {0}".format(totals[6]/iterations))
    print("---------------------------------------------")

colors = ['b', 'c', 'y', 'm', 'r']
import matplotlib.pyplot as plt

r1 = plt.scatter(x, results[0], marker='x', color=colors[0])
r2 = plt.scatter(x, results[1], marker='x', color=colors[1])
r3 = plt.scatter(x, results[2], marker='x', color=colors[2])
r4 = plt.scatter(x, results[3], marker='x', color=colors[3])
r5 = plt.scatter(x, results[4], marker='x', color=colors[4])
r6 = plt.scatter(x, results[5], marker='o', color=colors[0])
r7 = plt.scatter(x, results[6], marker='o', color=colors[1])

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
