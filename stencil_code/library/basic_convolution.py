from __future__ import print_function

import numpy
from stencil_code.stencil_kernel import Stencil
from stencil_code.neighborhood import Neighborhood


class ConvolutionFilter(Stencil):
    """
    basic filter requires user to pass in a matrix of coefficients. the
    dimensions of this convolution_array define the stencil neighborhood
    This should be a foundation class for example stencils such as the laplacians
    and jacobi stencils
    """
    def __init__(self, convolution_array=None, backend='ocl'):
        self.convolution_array = convolution_array
        neighbors, coefficients, _ = Neighborhood.compute_from_indices(convolution_array)
        self.neighbor_to_coefficient = dict(zip(neighbors, coefficients))
        self.coefficients = numpy.array(coefficients)
        super(ConvolutionFilter, self).__init__(
            neighborhoods=[neighbors], backend=backend, boundary_handling='copy'
        )

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

    def distance(self, x, y):
        d = tuple([x[i]-y[i] for i in range(len(x))])
        return self.neighbor_to_coefficient[d]

    def kernel(self, input_grid, coefficients, output_grid):
        for point in self.interior_points(output_grid):
            for n in self.neighbors(point, 0):
                output_grid[point] += input_grid[n] * self.distance(point, n)


if __name__ == '__main__':  # pragma no cover
    import logging
    logging.basicConfig(level=20)

    # in_grid = numpy.random.random([10, 5])
    in_grid = numpy.ones([8, 8, 16]).astype(numpy.float32)
    stencil = numpy.array(
        [
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            [
                [1, 1, 1],
                [1, 100, 1],
                [1, 1, 1],
            ],
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
        ]
    )

    ocl_convolve_filter = ConvolutionFilter(convolution_array=stencil, backend='ocl')
    ocl_out_grid = ocl_convolve_filter(in_grid)
    for i in [0, 1, 6, 7]:
        print("i {}".format(i))
        for r in ocl_out_grid[i]:
            for c in r:
                print("{:4.0f}".format(c), end="")
            print()
    exit(0)

    # in_grid = numpy.random.random([10, 5])
    in_grid = numpy.ones([32, 32]).astype(numpy.float32)
    stencil = numpy.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, -4, 1, 3],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 7, 1],
        ]
    )

    ocl_convolve_filter = ConvolutionFilter(convolution_array=stencil, backend='ocl')
    ocl_out_grid = ocl_convolve_filter(in_grid)
    for r in ocl_out_grid:
        for c in r:
            print("{:4.0f}".format(c), end="")
        print()
    exit(0)

    python_convolve_filter = ConvolutionFilter(convolution_array=stencil, backend='python')
    c_convolve_filter = ConvolutionFilter(convolution_array=stencil, backend='c')
    ocl_convolve_filter = ConvolutionFilter(convolution_array=stencil, backend='ocl')

    python_out_grid = python_convolve_filter(in_grid)
    c_out_grid = c_convolve_filter(in_grid)
    ocl_out_grid = ocl_convolve_filter(in_grid)
    print(python_out_grid)
    print()
    print(c_out_grid)
    print()
    print(ocl_out_grid)
