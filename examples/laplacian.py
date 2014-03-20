from numpy import *
from stencil_code.stencil_kernel import *
from stencil_code.stencil_grid import StencilGrid


import sys

alpha = 0.5
beta = 1.0


class LaplacianKernel(StencilKernel):
    def __init__(self, alpha, beta):
        super(LaplacianKernel, self).__init__()
        self.constants = {'alpha': alpha, 'beta': beta}

    def kernel(self, in_grid, out_grid):
        for x in in_grid.interior_points():
            out_grid[x] = alpha * in_grid[x]
            for y in in_grid.neighbors(x, 1):
                out_grid[x] += beta * in_grid[y]

nx = int(sys.argv[1])
ny = int(sys.argv[2])
nz = int(sys.argv[3])
input_grid = StencilGrid([nx, ny, nz])
output_grid = StencilGrid([nx, ny, nz])

for x in input_grid.interior_points():
    input_grid[x] = random.randint(nx * ny * nz)

laplacian = LaplacianKernel(alpha, beta)
for i in range(50):
    for x in input_grid.interior_points():
        input_grid[x] = random.randint(nx * ny * nz)
    laplacian.kernel(input_grid, output_grid)
