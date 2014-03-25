__author__ = 'Chick Markley'

from stencil_code.stencil_grid import StencilGrid
import timeit

sg = StencilGrid([1000, 1000, 1000])


def run_border():
    x = 0
    for bp in sg.border_points():
        x += 1
    print "got %d border points" % x


def run_boundary():
    x = 0
    for bp in sg.boundary_points():
        x += 1
    print "got %d border points" % x

if __name__ == '__main__':
    sg = StencilGrid([1000, 1000, 1000])
    print "border takes %s" % timeit.Timer(run_border).timeit(number=3)
    print "corner takes %s" % timeit.Timer(run_border).timeit(number=3)
    sg = StencilGrid([1000, 1000, 1000])
    print "boundary takes %s" % timeit.Timer(run_boundary).timeit(number=3)