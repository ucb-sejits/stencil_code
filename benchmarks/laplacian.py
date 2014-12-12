__author__ = 'chick'

import numpy

from benchmarks.stencil_benchmarker import StencilBenchmarker, StencilTest, PrimedStencilTest
from stencil_code.library.laplacian import LaplacianKernel


# import logging
# logging.basicConfig(level=20)

numpy.random.seed(1)


class LaplacianBenchmarker(StencilBenchmarker):
    def matrix_iterator(self):
        for height in [2**size for size in range(4, 5)]:
            for width in [2**size for size in range(5, 8)]:
                for depth in [2**size for size in range(self.min_power, self.max_power)]:
                    yield numpy.random.random([height, width, depth]).astype(numpy.float32)


if __name__ == '__main__':
    benchmarker = LaplacianBenchmarker(
        [
            # StencilTest("python", stencil=LaplacianKernel, "python"),
            StencilTest("unprimed-c", stencil=LaplacianKernel, backend='c'),
            PrimedStencilTest("primed-c", stencil=LaplacianKernel, backend='c'),
            StencilTest("unprimed-ocl", stencil=LaplacianKernel, backend='ocl'),
            PrimedStencilTest("primed-ocl", stencil=LaplacianKernel, backend='ocl'),
        ]
    )

    benchmarker.run()