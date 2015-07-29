from __future__ import print_function

__author__ = 'Chick Markley chick@eecs.berkeley.edu U.C. Berkeley'

"""
PDE heat flow simulation using a kernel.  Input is a 1d array where that is marched in time.

"""
# from __future__ import print_function
import numpy as np
import numpy.testing

from stencil_code.stencil_kernel import Stencil
from ctree.util import Timer
from scipy.integrate import odeint

NT = 5000
DT = 0.0001
NDIV = 50
MU = 0.5
SIGMA = 0.1
X0 = 1.0
X1 = 0.0
# width = 256
# height = 256
# time_steps = 16

N = NDIV + 1

ODEINT = False
VALIDATE = True
TIMEIT = True
NUMPY = True


class OneDHeatFlow(Stencil):
    """
    dx / dt = (x_{i+1} + x_{i-1} - 2 x_{i}) / dx^2

    """
    neighborhoods = [
        [(1,), (-1,)],  # coef is 1.0 / dx^2
        [(0,)],  # coef is -2 / dx^2
    ]


    def __init__(self, *args, **kwargs):
        """
        Params
        ------
        dx : float, optional
             Grid spacing (defaults to 1.0)
        """
        self.dx = kwargs.pop('dx', 1.0)
        self.n = kwargs.pop('dx', N)
        #print 'self.dx = ', self.dx
        super(OneDHeatFlow, self).__init__(*args, **kwargs)

    def kernel(self, in_grid, out_grid):
        a = 1.0 / (self.dx * self.dx)
        b = -2.0 / (self.dx * self.dx)
        for x in self.interior_points(out_grid):
            out_grid[x] = 0.0
            for y in self.neighbors(x, 0):
                out_grid[x] += a * in_grid[y]
            for y in self.neighbors(x, 1):
                out_grid[x] += b * in_grid[y]


class NumPyHeat(object):
    def __init__(self, dx=1.0, n=N):
        self.out_grid = np.zeros(n)
        self.coefs = np.array([1, -2, 1]) / dx ** 2

    def __call__(self, in_grid):
        self.out_grid[1:-1] = np.convolve(in_grid, self.coefs, 'valid')
        return self.out_grid


def np_kernel(in_grid, dx=1.0, coefs=np.array([1, -2, 1])):
    """Implement Laplace operator with NumPy.
    """
    out_grid = np.zeros_like(in_grid)
    out_grid[1:-1] = np.convolve(in_grid, coefs, 'valid')
    return out_grid / dx ** 2


if __name__ == '__main__':  # pragma: no cover
    # import logging
    # logging.basicConfig(level=20)
    kernel = OneDHeatFlow(backend='c')
    params = dict(dx=1.0 / NDIV, boundary="zero")
    #kernel = OneDHeatFlow(backend='c', **params)
    if NUMPY:
        py_kernel = NumPyHeat(dx=1.0 / NDIV, n=N)
    else:
        py_kernel = OneDHeatFlow(backend='python', **params)

    # Initialize solution
    x = np.linspace(0, 1, N)

    def dirichlet(q):
        q[0] = X0
        q[-1] = X1

    def gaussian(x, mu=0.5, sigma=0.1):
        return 1 / (2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)

    def init():
        q = np.zeros(N)
        dirichlet(q)
        return q

    def solve(q, kernel, dt=DT, nsteps=NT, boundary_condition=dirichlet):
        for i in xrange(nsteps):
            dq = kernel(q)
            q += dt * dq
            boundary_condition(q)
        return q

    with Timer() as t:
        q = init()
        if ODEINT:
            cq = odeint(lambda q, t: kernel(q), q, [0.0, (NT - 1) * DT])[-1]
        else:
            cq = solve(q, kernel)

    if TIMEIT:
        print("Specialized Time: %.03fs" % t.interval)

    if VALIDATE:
        with Timer() as py_time:
            q = init()
            pq = solve(q, py_kernel)

        # numpy.testing.assert_array_almost_equal(pq, cq)
        if TIMEIT:
            print("Python Time: %.03fs" % py_time.interval)

    # Print results
    for xi, yi in zip(x, cq):
        print('{:15.8f} {:15.8f}'.format(xi, yi))
