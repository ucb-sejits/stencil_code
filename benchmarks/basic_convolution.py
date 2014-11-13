from __future__ import print_function

import numpy
from stencil_code.stencil_kernel2 import Stencil
from stencil_code.neighborhood import Neighborhood


class ConvolutionFilter(Stencil):
    def __init__(self, convolution_array=None, backend='ocl'):
        neighbors, self.coefficients, _ = Neighborhood.compute_from_indices(convolution_array)
        super(ConvolutionFilter, self).__init__(neighborhoods=[neighbors], backend=backend)

    @staticmethod
    def clamped_add_tuple(point1, point2, grid):
        def clamp(d1_d2_max):
            return min(max(d1_d2_max[0] + d1_d2_max[1], 0), d1_d2_max[2]-1)

        return tuple(map(clamp, zip(point1, point2, grid.shape)))

    def __call__(self, *args, **kwargs):
        """
        We had to override __call__ here because the kernel currently cannot directly reference
        the smoothing arrays as class members so we must add them here to the call so
        the Stencil's __call__ can have access to them
        :param args:
        :param kwargs:
        :return:
        """
        new_args = [
            args[0],
            self.coefficients,
        ]
        return super(ConvolutionFilter, self).__call__(*new_args, **kwargs)

    def kernel(self, input_grid, coefficients, output_grid):
        for point in self.interior_points(output_grid):
            x = 0
            for n in self.neighbors(point, 0):
                output_grid[point] += input_grid[n] * coefficients[x]
                x += 1

        return output_grid


def main():
    # in_grid = numpy.random.random([10, 5])
    in_grid = numpy.ones([32, 32])
    stencil = numpy.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, -4, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )

    convolve_filter = ConvolutionFilter(convolution_array=stencil, backend='python')

    out_grid = convolve_filter(in_grid)
    print(out_grid)


if __name__ == '__main__':
    main()