from __future__ import print_function

import numpy as np
from stencil_code.stencil_kernel import Stencil, MultiConvolutionStencilKernel
from stencil_code.neighborhood import Neighborhood


class MultiConvolutionFilter(MultiConvolutionStencilKernel):
    """
    basic filter requires user to pass in a matrix of coefficients. the
    dimensions of this convolution_array define the stencil neighborhood
    This should be a foundation class for example stencils such as the
    laplacians and jacobi stencils
    """
    def __init__(self, convolution_arrays=None, stride=1, backend='ocl'):
        self.convolution_arrays = convolution_arrays
        self.num_convolutions = len(convolution_arrays)
        neighbors_list = []
        coefficients_list = []
        self.neighbor_sizes = []

        for convolution_array in convolution_arrays:
            neighbors, coefficients, _ = \
                Neighborhood.compute_from_indices(convolution_array)
            neighbors_list.append(neighbors)
            coefficients_list.append(coefficients)
            self.neighbor_sizes.append(len(neighbors))

        self.stride = stride
        self.my_neighbors = np.array(neighbors_list)  # this is not safe, doesn't consider boundaries yet
        self.coefficients = np.array(coefficients_list)
        super(MultiConvolutionFilter, self).__init__(
            neighborhoods=neighbors_list, backend=backend, boundary_handling='zero'
        )
        self.specializer.num_convolutions = self.num_convolutions

    def __call__(self, *args, **kwargs):
        """
        We had to override __call__ here because the kernel currently cannot
        directly reference the smoothing arrays as class members so we must add
        them here to the call so the Stencil's __call__ can have access to them
        :param args:
        :param kwargs:
        :return:
        """
        new_args = [
            args[0],
            self.coefficients,
        ]
        return super(MultiConvolutionFilter, self).__call__(*new_args, **kwargs)

    def multi_points(self, point):
        channel = point[0]
        for conv_id in range(self.num_convolutions):
            neighbor_count = 0
            for neighbor in self.neighbors(point, conv_id):
                input_index = point
                output_index = point[1:]
                # self.coefficients should be flattened
                coefficient = self.coefficients[conv_id][channel][neighbor_count]
                yield input_index, output_index, coefficient
                neighbor_count += 1

    def kernel(self, input_grid, coefficients, output_grid):
        for point in self.interior_points(input_grid, stride=self.stride):
            for input_index, output_index, coefficient in self.multi_points(point):
                output_grid[output_index] += input_grid[input_index] * coefficient


if __name__ == '__main__':  # pragma no cover
    import logging
    logging.basicConfig(level=20)

    bottom = np.random.rand(3, 227, 227).astype(np.float32) * 255.0
    top = np.zeros((96, 227, 227)).astype(np.float32)
    weights = np.random.rand(96, 3, 5, 5).astype(np.float32) * 2.0 - 1.0

    in_rows = 20
    in_cols = 20
    num_conv = 5
    conv_rows = 5
    conv_cols = 5
    in_grid = np.ones([in_rows, in_cols]).astype(np.float32)
    base_stencil = np.ones((conv_rows, conv_cols)).astype(np.float32)
    stencil = np.empty((num_conv, conv_rows, conv_cols)).astype(np.float32)
    for i in range(in_rows):
        for j in range(in_cols):
            in_grid[i][j] = j
    for i in range(num_conv):
        for j in range(conv_rows):
            for k in range(conv_cols):
                # stencil[i][j][k] = base_stencil[j][k]
                stencil[i][j][k] = 5#float((i+1)*(j+1))

    ocl_convolve_filter = MultiConvolutionFilter(convolution_arrays=stencil,
                                                 backend='ocl')
    ocl_out_grid = ocl_convolve_filter(in_grid)

    for r in in_grid:
        for c in r:
            print("{:4.0f} ".format(c), end="")
        print()

    for conv in range(ocl_convolve_filter.specializer.num_convolutions):
        print("convolution {}".format(conv))
        for r in ocl_out_grid[conv]:
            for c in r:
                print("{:4.0f} ".format(c), end="")
            print()
    exit(0)

