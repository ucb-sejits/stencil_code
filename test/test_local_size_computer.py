from __future__ import print_function
__author__ = 'chick'

import unittest

from stencil_code.backend.local_size_computer import product, LocalSizeComputer


class MockDevice(object):
    def __init__(self, max_work_group_size=512, max_work_item_sizes=None,
                 max_compute_units=40):
        self.max_work_group_size = max_work_group_size
        self.max_work_item_sizes = max_work_item_sizes if max_work_item_sizes is not None else [512, 512, 512]
        self.max_compute_units = max_compute_units

MockCPU = MockDevice(1024, [1024, 1, 1], 8)
MockIrisPro = MockDevice(512, [512, 512, 512], 40)


class TestLocalSizeComputer(unittest.TestCase):
    def _are_lists_equal(self, list1, list2):
        self.assertEqual(sorted(list1), sorted(list2))

    def _are_lists_unequal(self, list1, list2):
        self.assertNotEqual(sorted(list1), sorted(list2))

    def test_prod(self):
        self.assertTrue(product([1]) == 1)
        self.assertTrue(product([2, 3, 4]) == 24)

    def test_compute_local_group_size_1d(self):
        # chooses the minimum of shape / 2 and max local group size
        self.assertTrue(
            LocalSizeComputer([100], MockIrisPro).compute_local_size_thin() == (50,),
            "when smaller than work group divide by 2"
        )
        # print("ls1d {}".format(tools.compute_local_size([1000])))

        self.assertTrue(
            LocalSizeComputer([1000], MockIrisPro).compute_local_size_thin() == (500,),
            "when smaller than work group divide by 2"
        )

        self.assertTrue(
            LocalSizeComputer([10000], MockIrisPro).compute_local_size_thin() == (512,),
            "when smaller than work group divide by 2"
        )

    def test_compute_local_group_size_2d(self):
        # this device looks like a 2014 Iris Pro
        # the following numbers have not yet been tested for optimality
        # they are mostly a product of a desire for consistency and
        # minimization of unused cycles
        # TODO: fix both generator and number to reflect optimality

        test_cases = [
            [[1, 101], (1, 101)],
            [[101, 1], (101, 1)],
            [[512, 101], (19, 26)],
            [[512, 513], (1, 257)],
            [[300, 1025], (1, 342)],
            [[5120, 32], (16, 32)],
            [[5120011, 320001], (1, 512)],
            [[102, 7], (102, 4)]
        ]

        for grid_size, predicted_local_size in test_cases:
            local_size = LocalSizeComputer(grid_size, MockIrisPro).compute_local_size_thin()
            self.assertListEqual(list(local_size), list(predicted_local_size))

    def test_compute_local_group_size_3d(self):
        test_cases = [
            [[1, 1, 101], (1, 1, 1)],
            [[1, 101, 1], (1, 1, 1)],
            [[101, 1, 1], (101, 1, 1)],
            [[100, 512, 101], (100, 1, 1)]
        ]

        for grid_size, predicted_local_size in test_cases:
            local_size = LocalSizeComputer(grid_size, MockCPU).compute_local_size_thin()
            self.assertListEqual(list(local_size), list(predicted_local_size))

        test_cases = [
            [[1, 1, 101], (1, 1, 101)],
            [[1, 101, 1], (1, 101, 1)],
            [[101, 1, 1], (101, 1, 1)],
            [[100, 512, 101], (1, 19, 26)]
        ]

        for grid_size, predicted_local_size in test_cases:
            local_size = LocalSizeComputer(grid_size, MockIrisPro).compute_local_size_thin()
            self.assertListEqual(list(local_size), list(predicted_local_size))

    def test_local_size_computer_bulky(self):
        test_grid_shape = [4, 128]
        lsc = LocalSizeComputer(test_grid_shape, MockIrisPro)
        local_size = lsc.compute_local_size_bulky()
        print("{} ls {}".format(test_grid_shape, local_size))

        sizes = [
            [[4], (4,), (4,)],
            [[5], (5,), (5,)],
            [[255], (255,), (255,)],
            [[1023], (1023,), (341,)],
            [[1024], (1024,), (512,)],
            [[1025], (205,), (205,)],

            [[1, 4], (1, 1), (1, 4)],
            [[4, 1], (4, 1), (4, 1)],
            [[4, 4], (4, 1), (4, 4)],
            [[4, 128], (4, 1), (4, 128)],
            [[128, 4], (128, 1), (32, 4)],
            [[128, 7], (128, 1), (32, 7)],
            [[128, 128], (128, 1), (16, 32)],

            [[4, 4, 4], (4, 1, 1), (4, 4, 4)],
            [[4, 4, 512], (4, 1, 1), (4, 4, 32)],
            [[512, 512, 4], (512, 1, 1), (8, 8, 4)],
            [[512, 512, 512], (512, 1, 1), (8, 8, 8)],

            [[3, 3, 666], (3, 1, 1), (3, 3, 37)],
            [[99, 99, 99], (99, 1, 1), (3, 11, 11)],
        ]
        for grid_shape, expected_cpu_local_size, expected_gpu_local_size in sizes:
            print("size {!s:16s}".format(grid_shape), end="")
            cpu_local_size = LocalSizeComputer(grid_shape, MockCPU).compute_local_size_bulky()
            gpu_local_size = LocalSizeComputer(grid_shape, MockIrisPro).compute_local_size_bulky()

            print(" d0 cpu local_size {!s:15s} gpu local_size {!s:15s}".format(cpu_local_size, gpu_local_size))

            self.assertListEqual(list(cpu_local_size), list(expected_cpu_local_size))
            self.assertListEqual(list(gpu_local_size), list(expected_gpu_local_size))
