from stencil_code.stencil_kernel import Stencil
import numpy
import numpy.testing


class DiagnosticStencil(Stencil):
    # neighborhoods = [[(0, -1), (0, 1)], [(-1, 0), (1, 0)]]
    neighborhoods = [
        [(0, -1)], [(0, 1)], [(-1, 0)], [(1, 0)]
    ]

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            for y in self.neighbors(x, 0):
                out_grid[x] += 2 * in_grid[y]
            for y in self.neighbors(x, 1):
                out_grid[x] += 4 * in_grid[y]
            for y in self.neighbors(x, 2):
                out_grid[x] += 8 * in_grid[y]
            for y in self.neighbors(x, 3):
                    out_grid[x] += 16 * in_grid[y]


if __name__ == '__main__':  # pragma no cover
    import logging
    logging.basicConfig(level=20)

    height = 2**12
    width = 2**12
    # in_img = numpy.ones((height, height)).astype(numpy.float32)
    in_img = numpy.random.rand(height, height).astype(numpy.int32) * 1000

    # ocl_stencil = DiagnosticStencil(backend='ocl', boundary_handling='copy')
    python_stencil = DiagnosticStencil(backend='c', boundary_handling='copy')
    filter = numpy.array([
        [0.0, 16.0, 0.0],
        [2.0, 0.0, 4.0],
        [0.0, 8.0, 0.0]
    ]).astype(numpy.int32)
    from scipy.ndimage.filters import convolve

    for i in range(10):
        # ocl_out = ocl_stencil(in_img)
        python_out = python_stencil(in_img)

        test_out = convolve(in_img, filter)
        #
        # print(ocl_out)
        #
        # for index1 in range(height):
        #     for index2 in range(width):
        #         p = (index1, index2)
        #         if int(ocl_out[p]*1000) != int(python_out[p]*1000):
        #             print("{} {} != {}".format(p, ocl_out[p], python_out[p]))
        #
        numpy.testing.assert_allclose(test_out[1:-1, 1:-1], python_out[1:-1, 1:-1])
