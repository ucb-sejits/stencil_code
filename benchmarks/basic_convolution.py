from __future__ import print_function
import numpy as np
from stencil_code.neighborhood import Neighborhood
# from stencil_code.stencil_kernel import StencilKernel

stencil = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, -4, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
)


class ConvolutionFilter(object):
    def __init__(self, convolution_array):
        self.neighbors, self.coefficients, self.halo = Neighborhood.compute_from_indices(convolution_array)

    @staticmethod
    def clamped_add_tuple(point1, point2, grid):
        def clamp(d1_d2_max):
            return min(max(d1_d2_max[0] + d1_d2_max[1], 0), d1_d2_max[2]-1)

        return tuple(map(clamp, zip(point1, point2, grid.shape)))

    def all_points(self, grid):
        iterator = np.nditer(grid, flags=['multi_index'])
        for _ in iterator:
            yield iterator.multi_index

    def neighbor_coefficient_iterator(self, grid, point):
        print(self.neighbors)
        for neighbor_index, neighbor_offset in enumerate(self.neighbors):
            print("{} {} + {}".format(neighbor_index, point, neighbor_offset))
            yield ConvolutionFilter.clamped_add_tuple(point, neighbor_offset, grid), self.coefficients[neighbor_index]

    def kernel(self, input_grid):
        output_grid = np.empty_like(input_grid)
        for point in self.all_points(output_grid):
            for neighbor_point, coefficient in self.neighbor_coefficient_iterator(input_grid, point):
                output_grid[point] += input_grid[neighbor_point] * coefficient

        return output_grid


def main():
    # in_grid = np.random.random([10, 5])
    in_grid = np.fromfunction(lambda x, y: x, [10, 4])

    convolve_filter = ConvolutionFilter(stencil)

    out_grid = convolve_filter.kernel(in_grid)
    print(out_grid)


if __name__ == '__main__':
    main()