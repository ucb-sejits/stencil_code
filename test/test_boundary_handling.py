import unittest
import numpy
from stencil_code.neighborhood import Neighborhood

from stencil_code.stencil_exception import StencilException
from stencil_code.stencil_kernel2 import Stencil


class TestBoundaryHandling(unittest.TestCase):
    def test_python_clamping(self):
        class Clamper(Stencil):
            neighborhoods = [Neighborhood.von_neuman_neighborhood(radius=1, dim=2)]

            def kernel(self, in_grid, out_grid):
                for p in self.interior_points(out_grid):
                    for n in self.neighbors(p, 0):
                        out_grid[p] += in_grid[n]

        clamper = Clamper(backend='python', boundary_handling='clamp')
        in_grid = numpy.ones([10, 10])

        clamper.current_shape = in_grid.shape
        p = (9, 9)
        print("{}.neighbors {}".format(p, [x for x in clamper.neighbors(p, 0)]))
