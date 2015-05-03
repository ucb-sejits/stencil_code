from __future__ import print_function
from ctree.c.nodes import BinaryOp, Add, Mul, SymbolRef, Constant

import numpy
from stencil_code.stencil_exception import StencilException
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

        # Am I using this?
        self.neighbor_to_coefficient = []
        for n in range(len(convolution_arrays)):
            self.neighbor_to_coefficient += dict(zip(neighbors_list[n], coefficients_list[n]))

        self.stride = stride
        self.my_neighbors = numpy.array(neighbors_list)  # this is not safe, doesn't consider boundaries yet
        self.coefficients = numpy.array(coefficients_list)
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

    # def kernel(self, input_grid, coefficients, output_grid):
    #     for conv_id in range(self.num_convolutions):
    #         my_grid = output_grid[conv_id]
    #         for point in self.interior_points(my_grid, stride=self.stride):
    #             for n in range(self.get_neighbor_sizes(conv_id)):
    #                 new_out = (conv_id,) + point
    #                 new_in = (conv_id,) + self.get_neighbor_point(point, conv_id, n)
    #                 # new_coef_ind = (conv_id, n)
    #                 output_grid[new_out] += input_grid[new_in] * self.coefficients[conv_id][n]

    def neighbor_and_coefficient(self, point):
        """
        :param point: a tuple indexes into one point of the input grid
        :return:
        """
        for conv_id in range(self.num_convolutions):
            # neighbor_count = 0
            # for neighbor in self.neighbors(point, conv_id):
            #     input_index = point
            #     output_index = (conv_id,) + neighbor
            #     coefficient = self.coefficients[(conv_id, neighbor_count)]
            #     yield input_index, output_index, coefficient
            #     neighbor_count += 1
            try:
                if self.is_clamped and self.current_shape is not None:
                    for neighbor in self.neighborhood_definition[conv_id]:
                        yield tuple(map(
                            lambda dim: Stencil.clamp(point[dim]+neighbor[dim], 0,
                                                      self.current_shape[dim]),
                            range(len(point))))
                else:
                    for neighbor in self.neighborhood_definition[conv_id]:
                        yield tuple(map(lambda a, b: a+b, list(point),
                                        list(neighbor)))

            except IndexError:
                raise StencilException(
                    "Undefined neighborhood identifier {} this stencil has \
                        {}".format(conv_id, len(self.neighborhood_definition))) #??

    def orig_interior_points(self, x, stride=1):
        """
        this is a copy of interior points, but need to define new type of node for this kind of loop
        :param x:
        :param stride:
        :return:
        """
        if self.is_clamped:
            self.current_shape = x.shape
            dims = (range(0, dim, stride) for dim in x.shape)
        elif self.is_copied:
            dims = (range(self.ghost_depth[index], dim -
                          self.ghost_depth[index], stride) for index, dim in
                    enumerate(x.shape))
        else:
            dims = (range(self.ghost_depth[index], dim -
                          self.ghost_depth[index], stride) for index, dim in
                    enumerate(x.shape))

        for item in itertools.product(*dims):
            yield tuple(item)

    def kernel(self, input_grid, coefficients, output_grid):
        # for point in self.interior_points(input_grid, stride=self.stride):
        for point in self.interior_points(input_grid, stride=self.stride):
            for input_index, output_index, coefficient in self.neighbor_and_coefficient(point):
                output_grid[output_index] += input_grid[input_index] * coefficient
                # output_grid[conv_id *input_size + flatten(point)]

                # Add((Mult(SymbolRef("conv_id"), Constant(17))), self.index_target_dict['point'])


if __name__ == '__main__':  # pragma no cover
    import logging
    logging.basicConfig(level=20)

    # in_grid = numpy.random.random([10, 5])
    # in_grid = numpy.array([numpy.ones([8, 8]).astype(numpy.float32), numpy.ones([8, 8]).astype(numpy.float32)])
    in_grid = numpy.ones([8, 8, 16]).astype(numpy.float32)
    # stencil = numpy.array(
    #     [
    #         [
    #             [1.2, 1.5, 1.1],
    #             [6.2, 1.3, 4.2],
    #             [1.1, 3.1, 2.4],
    #         ],
    #         [
    #             [1.5, 3.1, 4.1],
    #             [1.2, 100, 1.6],
    #             [1.7, 2.7, 3.4],
    #         ],
    #         [
    #             [1.3, 4.1, 5.2],
    #             [1.5, 2.0, 3.5],
    #             [3.2, 6.1, 7.8],
    #         ],
    #     ]
    # )
    stencil = numpy.array(
        [
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
            ],
            [
                [
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                ],
                [
                    [2, 2, 2],
                    [2, 200, 2],
                    [2, 2, 2],
                ],
                [
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                ],
            ]
        ]
    )
    # stencil = numpy.array(  # TODO: make all these floats
    #     [
    #         [
    #             [1, 1, 1],
    #             [1, 100, 1],
    #             [1, 1, 1],
    #         ],
    #         # [
    #         #     [2, 2, 2],
    #         #     [2, 200, 2],
    #         #     [2, 2, 2],
    #         # ],
    #         # [
    #         #     [1, 1, 1],
    #         #     [1, 100, 1],
    #         #     [1, 1, 1],
    #         # ],
    #         # [
    #         #     [2, 2, 2],
    #         #     [2, 200, 2],
    #         #     [2, 2, 2],
    #         # ]
    #     ]
    # )

    ocl_convolve_filter = MultiConvolutionFilter(convolution_arrays=stencil,
                                                 backend='ocl')
    ocl_out_grid = ocl_convolve_filter(in_grid)
    print(ocl_out_grid)
    print(ocl_out_grid.shape)
    # for i in [0, 1, 6, 7]:
    # for conv in range(ocl_convolve_filter.num_convolutions):
    #     print("convolution {}".format(conv))
    #     # for conv in range(ocl_convolve_filter.num_convolutions):
    #     for r in ocl_out_grid[conv]:
    #         for c in r:
    #             print("{:4.0f}".format(c), end="")
    #         print()
    exit(0)

    # # in_grid = numpy.random.random([10, 5])
    # in_grid = numpy.ones([32, 32]).astype(numpy.float32)
    # stencil = numpy.array(
    #     [
    #         [1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1],
    #         [1, 1, -4, 1, 3],
    #         [1, 1, 1, 1, 1],
    #         [1, 1, 1, 7, 1],
    #     ]
    # )
    #
    # ocl_convolve_filter = MultiConvolutionFilter(convolution_arrays=stencil,
    #                                              backend='ocl')
    # ocl_out_grid = ocl_convolve_filter(in_grid)
    # for r in ocl_out_grid:
    #     for c in r:
    #         print("{:4.0f}".format(c), end="")
    #     print()
    # exit(0)
    #
    # python_convolve_filter = MultiConvolutionFilter(convolution_arrays=stencil,
    #                                                 backend='python')
    # c_convolve_filter = MultiConvolutionFilter(convolution_arrays=stencil,
    #                                            backend='c')
    # ocl_convolve_filter = MultiConvolutionFilter(convolution_arrays=stencil,
    #                                              backend='ocl')
    #
    # python_out_grid = python_convolve_filter(in_grid)
    # c_out_grid = c_convolve_filter(in_grid)
    # ocl_out_grid = ocl_convolve_filter(in_grid)
    # print(python_out_grid)
    # print()
    # print(c_out_grid)
    # print()
    # print(ocl_out_grid)
