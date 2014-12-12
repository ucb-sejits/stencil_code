from __future__ import print_function

from scipy.ndimage import convolve
import numpy
from stencil_code.library.basic_convolution import ConvolutionFilter
from ctree.util import Timer


if __name__ == '__main__':
    import sys

    include_omp = True if len(sys.argv) > 1 and sys.argv[1] is not '-omp' else False

    # we will use the following stencil with the specializers and with
    # scipy convolve
    stencil = numpy.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, -4, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )

    class Convolution(ConvolutionFilter):
        def __init__(self, backend):
            super(Convolution, self).__init__(stencil, backend=backend)

    x = []
    iterations = 10
    results = [[] for _ in range(7)]
    totals = [0.0 for _ in range(7)]

    for width in (2**x for x in range(10, 14)):
        height = width

        input_image = numpy.random.random([width, height]).astype(numpy.float32)
        output_image = numpy.empty_like(input_image)

        c_kernel = Convolution(backend='c')
        out_grid = c_kernel(input_image)

        ocl_kernel = Convolution(backend='ocl')
        out_grid = ocl_kernel(input_image)

        if include_omp:
            omp_kernel = Convolution(backend='omp')
            out_grid = omp_kernel(input_image)
        else:
            omp_kernel = None

        for _ in range(iterations):
            with Timer() as t0:
                out_image = convolve(input_image, stencil, mode='constant', cval=0.0)
            totals[0] += t0.interval
            results[0].append(t0.interval)
            x.append(width)

            with Timer() as t1:
                Convolution(backend='c')(input_image)
            totals[1] += t1.interval
            results[1].append(t1.interval)

            with Timer() as t2:
                c_kernel(input_image)
            totals[2] += t2.interval
            results[2].append(t2.interval)

            if include_omp:
                with Timer() as t3:
                    Convolution(backend='omp')(input_image)
                totals[3] += t3.interval
                results[3].append(t3.interval)

                with Timer() as t4:
                    omp_kernel(input_image)
                totals[4] += t4.interval
                results[4].append(t4.interval)

            with Timer() as t5:
                Convolution(backend='ocl')(input_image)
            totals[5] += t5.interval
            results[5].append(t5.interval)

            with Timer() as t6:
                ocl_kernel(input_image)
            totals[6] += t6.interval
            results[6].append(t6.interval)

        print("---------- Results for dim {0}x{1} ----------".format(width, height))
        print("Numpy convolve avg: {0}".format(totals[0]/iterations))
        print("Specialized C with compile time avg: {0}".format(totals[1]/iterations))
        print("Specialized C time avg without compile {0}".format(totals[2]/iterations))
        if include_omp:
            print("Specialized OpenMP with compile time avg: {0}".format(totals[3]/iterations))
            print("Specialized OpenMP time avg without compile {0}".format(totals[4]/iterations))
        print("Specialized OpenCL with compile time avg: {0}".format(totals[5]/iterations))
        print("Specialized OpenCL time avg without compile {0}".format(totals[6]/iterations))
        print("---------------------------------------------")

    colors = ['b', 'c', 'y', 'm', 'r']
    import matplotlib.pyplot as plt

    r1 = plt.scatter(x, results[0], marker='x', color=colors[0])
    r2 = plt.scatter(x, results[1], marker='x', color=colors[1])
    r3 = plt.scatter(x, results[2], marker='x', color=colors[2])
    if include_omp:
        r4 = plt.scatter(x, results[3], marker='x', color=colors[3])
        r5 = plt.scatter(x, results[4], marker='x', color=colors[4])
    else:
        r4 = r5 = None
    r6 = plt.scatter(x, results[5], marker='o', color=colors[0])
    r7 = plt.scatter(x, results[6], marker='o', color=colors[1])

    if include_omp:
        plt.legend((r1, r2, r3, r4, r5, r6, r7),
                   (
                       'Numpy convolve', 'C with compile', 'C without compile',
                       'OpenMP with compile', 'OpenMp without compile', 'OpenCL with compile',
                       'OpenCL without compile'
                   ),
                   scatterpoints=1,
                   loc='lower left',
                   ncol=3,
                   fontsize=8)
    else:
        plt.legend((r1, r2, r3, r6, r7),
                   (
                       'Numpy convolve', 'C with compile', 'C without compile', 'OpenCL with compile',
                       'OpenCL without compile'
                   ),
                   scatterpoints=1,
                   loc='lower left',
                   ncol=3,
                   fontsize=8)
    plt.show()
