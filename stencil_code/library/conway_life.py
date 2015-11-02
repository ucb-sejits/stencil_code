"""
use a stencil kernel to compute successive generation of a conway life game
"""

from __future__ import print_function

import numpy as np
import copy
from stencil_code.neighborhood import Neighborhood
from stencil_code.stencil_kernel import Stencil


class ConwayKernel(Stencil):
    """
    in_img is the life board at time t
    out_img is the life board at time t+1
    new_state_map defines the output state for a cell
        first 9 indices are for all possible neighbor sum around dead cell,
        next 9 indices are for all possible neighbor sum around live cell
        value at index is the new cell state
    """
    def __init__(self, backend='c'):
        super(ConwayKernel, self).__init__(
            neighborhoods=[Neighborhood.moore_neighborhood()],
            backend=backend,
            should_unroll=False
        )

    def kernel(self, in_img, new_state_map, out_img):
        for x in self.interior_points(out_img):
            out_img[x] = in_img[x] * 9
            for y in self.neighbors(x, 0):
                out_img[x] += in_img[y]
            out_img[x] = new_state_map[int(out_img[x])]


class IteratedConwayKernel(Stencil):
    """
    in_img is the life board at time t
    out_img is the life board at time t+1
    new_state_map defines the output state for a cell
        first 9 indices are for all possible neighbor sum around dead cell,
        next 9 indices are for all possible neighbor sum around live cell
        value at index is the new cell state
    """
    def __init__(self, generations, backend='c'):
        self.generations = generations
        super(IteratedConwayKernel, self).__init__(
            neighborhoods=[Neighborhood.moore_neighborhood()],
            backend=backend,
            should_unroll=False
        )

    def kernel(self, in_img, new_state_map, out_img):
        for generation in range(self.generations):
            for x in self.interior_points(out_img):
                out_img[x] = in_img[x] * 9
                for y in self.neighbors(x, 0):
                    out_img[x] += in_img[y]
                out_img[x] = new_state_map[int(out_img[x])]


class GameRunner(object):
    def __init__(self, width, height, should_unroll=False):
        self.width = width
        self.height = height

        self.kernel = IteratedConwayKernel(7)
        self.kernel.should_unroll = should_unroll

        # create a stencil grid for t+1
        self.current_grid = np.zeros([height, width])
        all_neighbors = [(x, y) for x in range(-1, 2) for y in range(-1, 2)]
        all_neighbors.remove((0, 0))
        self.future_grid = copy.deepcopy(self.current_grid)  # this will be swapped to current after each iteration

        self.new_state_map = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])

    def run_game(self, generations=1):
        for generation in range(generations):
            self.future_grid = self.kernel(self.current_grid, self.new_state_map)
            self.current_grid, self.future_grid = self.future_grid, self.current_grid
            print("gen %s" % generation)

    def __call__(self, *args, **kwargs):
        self.run_game()

    def run(self):
        self.run_game()

    @staticmethod
    def render_grid(sk, msg="---"):
        """
        Simplistic render of a life board
        """
        print(msg)
        for h in range(sk.shape[0]):
            for w in range(sk.shape[1]):
                if sk[h][w] > 0:
                    print('*', end='')
                else:
                    print(' ', end='')
            print('')


if __name__ == '__main__':
    import sys
    import timeit

    parameters = len(sys.argv)

    width = 25 if parameters < 2 else int(sys.argv[1])
    height = 25 if parameters < 3 else int(sys.argv[2])
    generations = 1 if parameters < 4 else int(sys.argv[3])

    game_runner = GameRunner(width, height)

    game_runner.kernel = IteratedConwayKernel(7, 'python')
    print("average time for un-specialized %s" % timeit.timeit(stmt=game_runner, number=10))

    game_runner.kernel = IteratedConwayKernel(7)
    print("average time for specialized %s" % timeit.timeit(stmt=game_runner, number=10))
