import ctree
import math
from stencil_code.stencil_kernel import StencilKernel
from stencil_code.stencil_grid import StencilGrid
import numpy as np
import time

height = 50
stdev_d = 3
stdev_s = 70
radius = 1
width = 64 + radius*2


class Kernel(StencilKernel):
    def kernel(self, in_grid, out_grid):
        for x in out_grid.interior_points():
            for y in in_grid.neighbors(x, 1):
                out_grid[x] += in_grid[y]

kernel = Kernel()
kernel.should_unroll = False
out_grid1 = StencilGrid([width, width])
out_grid1.ghost_depth = radius
out_grid2 = StencilGrid([width, width])
out_grid2.ghost_depth = radius
in_grid = StencilGrid([width, width])
in_grid.ghost_depth = radius

for x in range(0, width):
    for y in range(0, width):
        in_grid.data[(x, y)] = 1.0

# for x in range(-radius, radius+1):
#     for y in range(-radius, radius+1):
#         in_grid.neighbor_definition[1].append((x, y))


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

# with Timer() as ocl_t:
Kernel(backend='ocl').kernel(in_grid, out_grid1)
exit()

with Timer() as omp_t:
    Kernel().kernel(in_grid, out_grid2)
np.testing.assert_array_almost_equal(out_grid1.data, out_grid2.data)
print("OCL version time: %.03fs" % ocl_t.interval)
print("OMP version time: %.03fs" % omp_t.interval)

# print(in_grid)
# print(out_grid)
