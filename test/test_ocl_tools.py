from __future__ import print_function
__author__ = 'chick'

import unittest

from stencil_code.backend.ocl_tools import product, LocalSizeComputer


class MockDevice(object):
    def __init__(self, max_work_group_size=512, max_work_item_sizes=None,
                 max_compute_units=40):
        self.max_work_group_size = max_work_group_size
        self.max_work_item_sizes = max_work_item_sizes if max_work_item_sizes is not None else [512, 512, 512]
        self.max_compute_units = max_compute_units

MockCPU = MockDevice(1024, [1024, 1, 1], 8)
MockIrisPro = MockDevice(512, [512, 512, 512], 40)


class TestOclTools(unittest.TestCase):
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
        lsc = LocalSizeComputer([1, 4], MockCPU)
        print("[1, 4] ls {}".format(lsc.compute_local_size_bulky()))
        print(lsc.dimension_processing_priority_key(0))
        print(lsc.dimension_processing_priority_key(1))

        sizes = [
            [[4], (4,), (4,)],
            [[5], (5,), (5,)],
            [[255], (255,), (255,)],
            [[1023], (1023,), (512,)],
            [[1024], (1024,), (512,)],
            [[1025], (1024,), (512,)],

            [[1, 4], (1, 1), (1, 4)],
            [[4, 1], (4, 1), (4, 1)],
            [[4, 4], (4, 1), (4, 4)],
            [[4, 128], (4, 1), (4, 85)],
            [[128, 4], (128, 1), (22, 4)],
            [[128, 7], (128, 1), (22, 7)],
            [[128, 128], (128, 1), (22, 23)],

            [[4, 4, 4], (4, 1, 1), (4, 4, 4)],
            [[4, 4, 512], (4, 1, 1), (4, 4, 14)],
            [[512, 512, 4], (512, 1, 1), (8, 8, 4)],
            [[512, 512, 512], (512, 1, 1), (8, 8, 8)],

            [[3, 3, 633], (3, 1, 1), (3, 3, 32)],
            [[99, 99, 99], (99, 1, 1), (8, 8, 8)],
        ]
        for grid_shape, cpu_local_size, gpu_local_size in sizes:
            print("size {:16}".format(grid_shape), end="")
            c1 = LocalSizeComputer(grid_shape, MockCPU).compute_local_size_bulky()
            c2 = LocalSizeComputer(grid_shape, MockIrisPro).compute_local_size_bulky()

            print(" d0 cpu local_size {:15} gpu local_size {:15}".format(c1, c2))

            self.assertListEqual(list(c1), list(cpu_local_size))
            self.assertListEqual(list(c2), list(gpu_local_size))
