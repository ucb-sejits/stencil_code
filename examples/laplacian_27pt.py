from __future__ import print_function

import numpy
import numpy.testing
from stencil_code.stencil_kernel2 import Stencil
from stencil_code.neighborhood import Neighborhood

import logging
logging.basicConfig(level=20)


class SpecializedLaplacian27(Stencil):
    neighborhoods = [
        Neighborhood.moore_neighborhood(radius=1, dim=3, include_origin=True),
    ]

    def distance(self, x, y):
        """
        override the StencilKernel distance and use manhattan distance
        """
        return sum([abs(x[i]-y[i]) for i in range(len(x))])

    def kernel(self, source_data, factors, output_data):
        """
        using distance above as index into coefficient array, perform stencil
        """
        for x in self.interior_points(output_data):
            for n in self.neighbors(x, 0):
                output_data[x] += factors[self.distance(x, n)] * source_data[n]


def laplacian_27pt(nx, ny, nz, alpha, beta, gamma, delta, source, destination):
    """
    An actual hand written 27 point laplacian function, found in the field
    """
    for k in range(2, nz - 1):
        for j in range(2, ny - 1):
            for i in range(2, nx - 1):
                destination[i, j, k] = alpha * source[i, j, k] + \
                    beta * (source[i + 1, j, k] + source[i - 1, j, k] +
                            source[i, j + 1, k] + source[i, j - 1, k] +
                            source[i, j, k + 1] + source[i, j, k - 1]) + \
                    gamma * (source[i - 1, j, k - 1] + source[i - 1, j - 1, k] +
                             source[i - 1, j + 1, k] + source[i - 1, j, k + 1] +
                             source[i, j - 1, k - 1] + source[i, j + 1, k - 1] +
                             source[i, j - 1, k + 1] + source[i, j + 1, k + 1] +
                             source[i + 1, j, k - 1] + source[i + 1, j - 1, k]) + \
                    delta * (source[i - 1, j - 1, k - 1] + source[i - 1, j + 1, k - 1] +
                             source[i - 1, j - 1, k + 1] + source[i - 1, j + 1, k + 1] +
                             source[i + 1, j - 1, k - 1] + source[i + 1, j + 1, k - 1] +
                             source[i + 1, j - 1, k + 1] + source[i + 1, j + 1, k + 1])


if __name__ == '__main__':
    import sys

    x_size = 32 if len(sys.argv) <= 1 else int(sys.argv[1])
    y_size = 32 if len(sys.argv) <= 2 else int(sys.argv[2])
    z_size = 32 if len(sys.argv) <= 3 else int(sys.argv[3])

    input_grid = numpy.random.random([x_size, y_size, z_size]).astype(numpy.float32)
    coefficients = numpy.array([1.0, 0.5, 0.25, 0.125]).astype(numpy.float32)
    ocl_laplacian = SpecializedLaplacian27(backend='ocl')
    c_laplacian = SpecializedLaplacian27(backend='c')
    python_laplacian = SpecializedLaplacian27(backend='python')

    ocl_output = ocl_laplacian(input_grid, coefficients)
    c_output = c_laplacian(input_grid, coefficients)
    python_output = python_laplacian(input_grid, coefficients)

    print("specialized ocl output[2][2][:] {}".format(ocl_output[2, 2, 2:z_size-1]))
    print("specialized c   output[2][2][:] {}".format(ocl_output[2, 2, 2:z_size-1]))
    numpy.testing.assert_array_almost_equal(ocl_output[2:x_size-1, 2:y_size-1, 2:z_size-1], c_output[2:x_size-1, 2:y_size-1, 2:z_size-1])
    numpy.testing.assert_array_almost_equal(ocl_output[2:x_size-1, 2:y_size-1, 2:z_size-1], python_output[2:x_size-1, 2:y_size-1, 2:z_size-1])

    # exit(1)

    out_grid = numpy.empty_like(input_grid)
    laplacian_27pt(x_size, y_size, z_size,
                   alpha=1.0, beta=0.5, gamma=0.25, delta=0.125,
                   source=input_grid, destination=out_grid)
    print("X"*120)
    print("python   output[2][2][:] {}".format(out_grid[2, 2, 2:z_size-1]))

    numpy.testing.assert_array_almost_equal(ocl_output[2:x_size-1, 2:y_size-1, 2:z_size-1], out_grid[2:x_size-1, 2:y_size-1, 2:z_size-1])