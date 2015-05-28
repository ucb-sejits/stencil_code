from __future__ import print_function

import numpy as np
from stencil_code.stencil_kernel import MultiConvolutionStencilKernel
from stencil_code.neighborhood import Neighborhood

# import numpy
# import pycuda.autoinit
# import pycuda.driver as cuda
# from pycuda import gpuarray
# from pycuda.compiler import SourceModule
# import scikits.cuda.cublas as cublas
# import string
import numpy as np

# im2col CUDA kernel
im2col_str = """
__global__ void im2col_gpu_kernel(const int n, const float* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    float* data_col) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += blockDim.x * gridDim.x) {
    int w_out = i % width_col;
    int h_index = i / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    float* data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    const float* data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
            data_im_ptr[i * width + j] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}
"""


def cuda_get_blocks(N):
    return (N + num_threads - 1) / num_threads


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
            current_coefficients = []
            for channel in convolution_array:
                neighbors, coefficients, _ = \
                    Neighborhood.compute_from_indices(channel)
                neighbors_list.append(neighbors)
                current_coefficients.append(coefficients)
                self.neighbor_sizes.append(len(neighbors))
            coefficients_list.append(current_coefficients)

        self.stride = stride
        self.my_neighbors = np.array(neighbors_list)  # this is not safe, doesn't consider boundaries yet
        self.coefficients = np.array(coefficients_list)
        super(MultiConvolutionFilter, self).__init__(
            neighborhoods=neighbors_list, backend=backend, boundary_handling='zero'
        )
        self.specializer.num_convolutions = self.num_convolutions
        # self.multiple_kernels = True

    def __call__(self, *args, **kwargs):
        """
        We had to override __call__ here because need to separate input into channels
        :param args:
        :param kwargs:
        :return:
        """
        new_args = [
            args[0][0],
            args[0][1],
            args[0][2]
        ]
        return super(MultiConvolutionFilter, self).__call__(*new_args, **kwargs)

    def multi_points(self, point):
        channel = point[0]
        for conv_id in range(self.num_convolutions):
            neighbor_count = 0
            for neighbor in self.neighbors(point, conv_id):
                input_index = point
                output_index = point[1:]
                coefficient = self.coefficients[conv_id][channel][neighbor_count]
                yield input_index, output_index, coefficient
                neighbor_count += 1

    def kernel(self, channel0, channel1, channel2, output_grid):
        for point in self.interior_points(channel0, stride=self.stride):
            for input_index, output_index, coefficient in self.multi_points(point):
                output_grid[output_index] += channel0[input_index] * coefficient
        for point in self.interior_points(channel1, stride=self.stride):
            for input_index, output_index, coefficient in self.multi_points(point):
                output_grid[output_index] += channel1[input_index] * coefficient
        for point in self.interior_points(channel2, stride=self.stride):
            for input_index, output_index, coefficient in self.multi_points(point):
                output_grid[output_index] += channel2[input_index] * coefficient
        # repeat above for each channel


if __name__ == '__main__':  # pragma no cover

    # cublas_handle = cublas.cublasCreate()
    #
    # module = SourceModule(im2col_str)
    # im2col = module.get_function('im2col_gpu_kernel')
    #
    # num_threads = 1024

    # Setup buffers
    input_height = 25
    input_width = 25
    num_conv = 2
    bottom = np.ones((3, input_height, input_width)).astype(np.float32) * 255.0
    # bottom = np.random.rand(3, input_height, input_width).astype(np.float32) * 255.0
    top = np.zeros((num_conv, input_height, input_width)).astype(np.float32)
    weights = np.ones((num_conv, 3, 5, 5)).astype(np.float32) * 2.0 - 1.0
    # weights = np.random.rand(num_conv, 3, 5, 5).astype(np.float32) * 2.0 - 1.0

    channels, height, width = bottom.shape
    padding = 2
    kernel_size = 5
    stride = 1
    height_col = (height + 2 * padding - kernel_size) / stride + 1
    width_col = (width + 2 * padding - kernel_size) / stride + 1

    # channels_col = channels * np.prod(weights.shape[2:])
    # col_buf = gpuarray.empty((channels_col, height_col, width_col), np.float32)
    # bottom_gpu = gpuarray.to_gpu(bottom)
    # weights_gpu = gpuarray.to_gpu(weights)
    # top_gpu = gpuarray.to_gpu(top)


    def conv(bottom, weights, top, kernel_size=np.uint32(5), padding=np.uint32(2),
             stride=np.uint32(1)):
        num_kernels = np.uint32(channels * height_col * width_col)
        m = np.int32(weights.shape[0])
        n = np.int32(np.prod(top.shape[1:]))
        k = np.int32(np.prod(weights.shape[1:]))
        im2col(num_kernels, bottom, np.uint32(height), np.uint32(width),
               kernel_size, kernel_size, padding, padding, stride, stride,
               np.uint32(height_col), np.uint32(width_col), col_buf,
               grid=(cuda_get_blocks(num_kernels), 1, 1),
               block=(num_threads, 1, 1))
        cublas.cublasSgemm(cublas_handle, 'n', 'n', n, m, k, np.float32(1.0),
                           col_buf.gpudata, n, weights.gpudata, k, np.float32(0.0),
                           top.gpudata, n)

    # conv(bottom_gpu, weights_gpu, top_gpu)

    import logging
    logging.basicConfig(level=20)

    ocl_convolve_filter = MultiConvolutionFilter(convolution_arrays=weights,
                                                 backend='ocl')
    ocl_out_grid = ocl_convolve_filter(bottom)

    new_out_grid = np.zeros((num_conv, input_height, input_width)).astype(np.float32)
    for conv in range(num_conv):
        for r in range(input_height):
            for c in range(input_width):
                new_out_grid[conv][r][c] = ocl_out_grid[c + r * input_width][conv]
    ocl_out_grid = new_out_grid

    # np.testing.assert_almost_equal(top_gpu.get(), ocl_out_grid, decimal=2)
    # print(top_gpu.get())
    print(ocl_out_grid)

    exit(0)

