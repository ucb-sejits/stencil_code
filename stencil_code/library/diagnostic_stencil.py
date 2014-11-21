from stencil_code.stencil_kernel2 import Stencil
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
                out_grid[x] += 2.0 * in_grid[y]
            for y in self.neighbors(x, 1):
                out_grid[x] += 4.0 * in_grid[y]
            for y in self.neighbors(x, 2):
                out_grid[x] += 8.0 * in_grid[y]
            for y in self.neighbors(x, 3):
                    out_grid[x] += 16.0 * in_grid[y]


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=20)

    height = 8
    width = 8
    in_img = numpy.ones([height, height]).astype(numpy.float32)

    ocl_stencil = DiagnosticStencil(backend='ocl', boundary_handling='copy')
    python_stencil = DiagnosticStencil(backend='python', boundary_handling='copy')

    ocl_out = ocl_stencil(in_img)
    python_out = python_stencil(in_img)

    print(ocl_out)

    for x in range(height):
        for y in range(width):
            p = (x, y)
            if int(ocl_out[p]*1000) != int(python_out[p]*1000):
                print("{} {} != {}".format(p, ocl_out[p], python_out[p]))

    numpy.testing.assert_array_almost_equal(ocl_out, python_out, decimal=4)
