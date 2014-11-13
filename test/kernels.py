from stencil_code.stencil_kernel2 import Stencil
from stencil_code.neighborhood import Neighborhood
import numpy
import math

# import logging
# logging.basicConfig(level=20)


class TwoDHeatFlow(Stencil):
    neighborhoods = [[(0, 0)], [
        (-1, 1),  (0, 1),  (1, 1),
        (-1, 0),  (0, 0),  (1, 0),
        (-1, -1), (-1, 0), (-1, 1)
    ]]

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(out_grid):
            out_grid[x] = in_grid[x]
            for y in self.neighbors(x, 0):
                out_grid[x] += 0.125 * in_grid[y]
            for z in self.neighbors(x, 1):
                out_grid[x] -= 0.25 * in_grid[z]


class LaplacianKernel(Stencil):
    neighborhoods = [Neighborhood.von_neuman_neighborhood(radius=1, dim=2)]

    def kernel(self, in_grid, out_grid):
        for x in self.interior_points(in_grid):
            out_grid[x] = 0.5 * in_grid[x]
            for y in self.neighbors(x, 0):
                out_grid[x] += in_grid[y]


class SpecializedLaplacian27(Stencil):
    """
    a 3d laplacian filter, overrides the default distance function used in looking up
    the factors
    """
    neighborhoods = [
        Neighborhood.moore_neighborhood(radius=1, dim=3, include_origin=False),
    ]
    constants = {'alpha': 0.5, 'beta': 1.0}

    def distance(self, x, y):
        """
        override the StencilKernel distance and use manhattan distance
        """
        return sum([abs(x[i]-y[i]) for i in range(len(x))])

    def kernel(self, source_data, output_data):
        """
        using distance above as index into coefficient array, perform stencil
        """
        for x in self.interior_points(output_data):
            output_data[x] = 0.5 * source_data[x]
            for n in self.neighbors(x, 0):
                output_data[x] += 1.0 * source_data[n]


class BetterBilateralFilter(Stencil):
    """
    An implementation of BilateralFilter that better encapsulates the
    internal tools used.
    """
    def __init__(self, sigma_d=1, sigma_i=70, backend='ocl'):
        """
        prepare the bilateral filter
        :param sigma_d: the smoothing associated with distance between points
        :param sigma_i: the smoothing factor associated with intensity difference between points
        :param backend:
        :return:
        """
        self.sigma_d = sigma_d
        self.sigma_i = sigma_i
        self.radius = 3 * self.sigma_d

        self.distance_lut = BetterBilateralFilter.gaussian(self.sigma_d, self.radius*2)
        self.intensity_lut = BetterBilateralFilter.gaussian(self.sigma_i, 256)

        super(BetterBilateralFilter, self).__init__(
            neighborhoods=[Neighborhood.moore_neighborhood(radius=self.radius, dim=2)],
            backend=backend,
            should_unroll=False
        )

    def __call__(self, *args, **kwargs):
        """
        We had to override __call__ here because the kernel currently cannot directly reference
        the smoothing arrays as class members so we must add them here to the call so
        the Stencil's __call__ can have access to them
        :param args:
        :param kwargs:
        :return:
        """
        new_args = [
            args[0],
            self.distance_lut,
            self.intensity_lut,
        ]
        return super(BetterBilateralFilter, self).__call__(*new_args, **kwargs)

    @staticmethod
    def gaussian(stdev, length):
        result = numpy.zeros(length).astype(numpy.float32)
        scale = 1.0/(stdev*math.sqrt(2.0*math.pi))
        divisor = -1.0 / (2.0 * stdev * stdev)
        for x in xrange(length):
            result[x] = scale * math.exp(float(x) * float(x) * divisor)
        return result

    def distance(self, x, y):
        return math.sqrt(sum([(x[i]-y[i])**2 for i in range(0, len(x))]))

    def kernel(self, in_img, filter_d, filter_s, out_img):
        for i in self.interior_points(out_img):
            for j in self.neighbors(i, 0):
                out_img[i] += in_img[j] * filter_d[int(self.distance(i, j))] *\
                    filter_s[abs(int(in_img[i] - in_img[j]))]

