import numpy
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import scikits.cuda.cublas as cublas
import string
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

cublas_handle = cublas.cublasCreate()

module = SourceModule(im2col_str)
im2col = module.get_function('im2col_gpu_kernel')

num_threads = 1024

# Setup buffers
bottom = np.random.rand(3, 227, 227).astype(np.float32) * 255.0
top = np.zeros((96, 227, 227)).astype(np.float32)
weights = np.random.rand(96, 3, 5, 5).astype(np.float32) * 2.0 - 1.0

channels, height, width = bottom.shape
padding = 2
kernel_size = 5
stride = 1
height_col = (height + 2 * padding - kernel_size) / stride + 1
width_col = (width + 2 * padding - kernel_size) / stride + 1

channels_col = channels * np.prod(weights.shape[2:])
col_buf = gpuarray.empty((channels_col, height_col, width_col), np.float32)
bottom_gpu = gpuarray.to_gpu(bottom)
weights_gpu = gpuarray.to_gpu(weights)
top_gpu = gpuarray.to_gpu(top)


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

conv(bottom_gpu, weights_gpu, top_gpu)
print(top_gpu.get())
