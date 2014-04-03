import ctree
import math
from stencil_code.stencil_kernel import StencilKernel
from stencil_code.stencil_grid import StencilGrid

width = 50
height = 50
stdev_d = 3
stdev_s = 70
radius = 1


class Kernel(StencilKernel):
    def kernel(self, in_grid, out_grid):
        for x in out_grid.interior_points():
            for y in in_grid.neighbors(x, 1):
                out_grid[x] += in_grid[y]

kernel = Kernel()
kernel.should_unroll = False
out_grid = StencilGrid([width, width])
out_grid.ghost_depth = radius
in_grid = StencilGrid([width, width])
in_grid.ghost_depth = radius

for x in range(0, width):
    for y in range(0, width):
        in_grid.data[(x, y)] = 1

Kernel(backend='ocl').kernel(in_grid, out_grid)
