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
                out_grid[x] += 2.0 * in_grid[y]
            for y in self.neighbors(x, 1):
                out_grid[x] += 4.0 * in_grid[y]
            for y in self.neighbors(x, 2):
                out_grid[x] += 8.0 * in_grid[y]
            for y in self.neighbors(x, 3):
                    out_grid[x] += 16.0 * in_grid[y]


if __name__ == '__main__':  # pragma no cover
    import logging
    logging.basicConfig(level=20)

    height = 8
    width = 8
    in_img = numpy.ones([height, height]).astype(numpy.float32)

    ocl_stencil = DiagnosticStencil(backend='ocl', boundary_handling='clamp')
    python_stencil = DiagnosticStencil(backend='c', boundary_handling='clamp')

    ocl_out = ocl_stencil(in_img)
    python_out = python_stencil(in_img)

    print(ocl_out)

    for index1 in range(height):
        for index2 in range(width):
            p = (index1, index2)
            if int(ocl_out[p]*1000) != int(python_out[p]*1000):
                print("{} {} != {}".format(p, ocl_out[p], python_out[p]))

    numpy.testing.assert_array_almost_equal(ocl_out, python_out, decimal=4)
