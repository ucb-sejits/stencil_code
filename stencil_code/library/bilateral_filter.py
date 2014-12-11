from __future__ import print_function

import numpy
from stencil_code.neighborhood import Neighborhood
from stencil_code.stencil_kernel import Stencil
import math


class BilateralFilter(Stencil):
    def __init__(self, radius=3, backend='ocl'):
        super(BilateralFilter, self).__init__(
            neighborhoods=[Neighborhood.moore_neighborhood(radius=radius, dim=2)],
            backend=backend,
            should_unroll=False
        )

    def distance(self, x, y):
        return math.sqrt(sum([(x[i]-y[i])**2 for i in range(0, len(x))]))

    def kernel(self, in_img, filter_d, filter_s, out_img):
        for i in self.interior_points(out_img):
            for j in self.neighbors(i, 0):
                out_img[i] += in_img[j] * filter_d[int(self.distance(i, j))] *\
                    filter_s[abs(int(in_img[i] - in_img[j]))]


def gaussian(stdev, length):
    result = numpy.zeros(length).astype(numpy.float32)
    scale = 1.0/(stdev*math.sqrt(2.0*math.pi))
    divisor = -1.0 / (2.0 * stdev * stdev)
    for x in range(length):
        result[x] = scale * math.exp(float(x) * float(x) * divisor)
    return result


if __name__ == '__main__':  # pragma no cover
    # import logging
    # logging.basicConfig(level=20)

    import sys

    width = int(sys.argv[2])
    height = int(sys.argv[3])
    image_in = open(sys.argv[1], 'rb')
    out_filename = "/dev/null" if len(sys.argv) < 5 else sys.argv[4]
    stdev_d = 3
    stdev_s = 70
    halo_size = stdev_d * 3

    pixels = map(ord, list(image_in.read(width * height)))  # Read in grayscale values
    intensity = float(sum(pixels))/len(pixels)
    print("intensity {}".format(intensity))

    ocl_bilateral_filter = BilateralFilter(halo_size, backend='ocl')
    c_bilateral_filter = BilateralFilter(halo_size, backend='c')

    # convert input stream into 2d array
    in_grid = numpy.zeros([height, width], numpy.float32)
    for index1 in range(height):
        for index2 in range(width):
            in_grid[index1, index2] = pixels[index1 * width + index2]

    gaussian1 = gaussian(stdev_d, halo_size*2)
    gaussian2 = gaussian(stdev_s, 256)

    ocl_out_grid = ocl_bilateral_filter(in_grid, gaussian1, gaussian2)
    c_out_grid = ocl_bilateral_filter(in_grid, gaussian1, gaussian2)

    for index1 in range(height):
        for index2 in range(width):
            pixels[index1 * width + index2] = (ocl_out_grid[index1, index2])

    # print(pixels)
    print("sum pix sum {} len {}".format(sum(pixels), len(pixels)))
    out_intensity = float(sum(pixels))/len(pixels)
    for index1 in range(0, len(pixels)):
        pixels[index1] = min(255, max(0, int(pixels[index1] * (intensity/out_intensity))))

    image_out = open(out_filename, 'wb')
    image_out.write(''.join(map(chr, pixels)))
