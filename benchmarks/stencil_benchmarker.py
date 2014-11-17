from __future__ import print_function

import abc
import numpy
from stencil_code.library.basic_convolution import ConvolutionFilter
from ctree.util import Timer


class StencilTest(object):
    """
    a test instance that supports runs with different test_ids
    """
    def __init__(self, name, stencil, backend, setup_args=()):
        self.name = name
        self.stencil_class = stencil
        self.setup_args = setup_args
        self.trial_times = dict()
        self.stencil_instance = None

    def setup(self):
        """a method that constructs a stencil, override to do something like implement a primed test"""
        pass

    def run(self, test_matrix):
        """
        override this method to change this
        :param test_matrix:
        :return:
        """
        stencil = self.stencil_class(*self.setup_args)
        return stencil(test_matrix)

    def run_trial(self, test_matrix, test_id=0):
        if test_id not in self.trial_times:
            self.trial_times[test_id] = []

        self.setup()

        with Timer() as timer:
            out_image = self.run(test_matrix)
        self.trial_times[test_id].append(timer.interval)

    def average_time(self, test_id):
        if test_id not in self.trial_times:
            raise Exception("Test {} does not contain data for test_id {}".format(self.name, test_id))

        if len(self.trial_times[test_id]) < 1:
            return 0

        return sum(self.trial_times[test_id]) / float(len(self.trial_times[test_id]))


class PrimedStencilTest(StencilTest):
    def setup(self):
        self.stencil_instance = self.stencil_class(*self.setup_args)

    def run(self, test_matrix):
        print("instance is {}".format(self.stencil_instance))
        return self.stencil_instance(test_matrix)


class StencilBenchmarker(object):
    """
    compares a number of individual tests against each other for a number of different input
    matrix sizes.
    """
    def __init__(self, tests_to_run, iterations=10):
        self.tests_to_run = tests_to_run
        self.iterations = iterations
        self.min_power = 10
        self.max_power = 12

    def matrix_iterator(self):
        """
        override this to produce other types of matrix sizes
        :return: a random matrix from an array of sizes
        """
        for width in [2**size for size in range(self.min_power, self.max_power)]:
            for height in [2**size for size in range(self.min_power, self.max_power)]:
                yield numpy.random.random([height, width]).astype(numpy.float32)

    def run(self):
        test_shapes = []
        for input_image in self.matrix_iterator():
            test_shapes.append(input_image.shape)
            for test in self.tests_to_run:
                for trials in xrange(self.iterations):
                    test.run_trial(input_image, input_image.shape)

        for test_shape in test_shapes:
            for test in self.tests_to_run:
                print("{},{},{}".format(
                    test_shape,
                    test.name,
                    test.average_time(test_shape)
                ))

