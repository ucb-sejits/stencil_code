import sys
import numpy
import math
import time

from stencil_code.library.better_bilateral_filter import BetterBilateralFilter

width = int(sys.argv[2])
height = int(sys.argv[3])
image_in = open(sys.argv[1], 'rb')
stdev_d = 3
stdev_s = 70
radius = stdev_d * 3


pixels = map(ord, list(image_in.read(width * height))) # Read in grayscale values
intensity = float(sum(pixels))/len(pixels)

kernel = BetterBilateralFilter()
in_grid = numpy.zeros([width, height]).astype(numpy.float32)

for x in range(0, width):
    for y in range(0, height):
        in_grid[(x, y)] = pixels[y * width + x]

out_grid = kernel(in_grid)

for x in range(0, width):
    for y in range(0,height):
        pixels[y * width + x] = out_grid[(x, y)]

out_intensity = float(sum(pixels))/len(pixels)
for i in range(0, len(pixels)):
    pixels[i] = min(255, max(0, int(pixels[i] * (intensity/out_intensity))))

image_out = open(sys.argv[4], 'wb')
image_out.write(''.join(map(chr, pixels)))
