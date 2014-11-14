import numpy
from stencil_code.neighborhood import Neighborhood
from stencil_code.stencil_kernel2 import Stencil
import math

# import logging
# logging.basicConfig(level=20)


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


if __name__ == '__main__':
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: {} raw_image_file width height [output_file] [sigma_d] [sigma_i]".format(
            os.path.basename(__file__)
        ))
        exit(1)

    width = int(sys.argv[2])
    height = int(sys.argv[3])
    image_in = open(sys.argv[1], 'rb')

    out_filename = "/dev/null" if len(sys.argv) < 5 else sys.argv[4]
    stdev_d = 1 if len(sys.argv) < 6 else sys.argv[5]
    stdev_s = 70 if len(sys.argv) < 7 else int(sys.argv[6])

    pixels = map(ord, list(image_in.read(width * height))) # Read in grayscale values
    intensity = float(sum(pixels))/len(pixels)
    print("intensity {}".format(intensity))

    ocl_bilateral_filter = BetterBilateralFilter(stdev_d, stdev_s, backend='ocl')
    c_bilateral_filter = BetterBilateralFilter(stdev_d, stdev_s, backend='c')

    # convert input stream into 2d array
    in_grid = numpy.zeros([height, width], numpy.float32)
    for i in xrange(height):
        for j in xrange(width):
            in_grid[i, j] = pixels[i * width + j]

    ocl_out_grid = ocl_bilateral_filter(in_grid)
    c_out_grid = ocl_bilateral_filter(in_grid)

    print("in  {}".format(in_grid))
    print("ocl_out {}".format(ocl_out_grid))
    print("c_out   {}".format(c_out_grid))

    for i in xrange(height):
        for j in xrange(width):
            pixels[i * width + j] = (ocl_out_grid[i, j])

    # print(pixels)
    print("sum pix sum {} len {}".format(sum(pixels), len(pixels)))
    out_intensity = float(sum(pixels))/len(pixels)
    for i in range(0, len(pixels)):
        pixels[i] = min(255, max(0, int(pixels[i] * (intensity/out_intensity))))

    image_out = open(out_filename, 'wb')
    image_out.write(''.join(map(chr, pixels)))
