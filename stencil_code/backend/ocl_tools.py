from __future__ import print_function
__author__ = 'chick'

from operator import mul
import numpy


def product(vector):
    return reduce(mul, vector, 1)


class OclTools(object):
    def __init__(self, device=None):
        if device is not None:
            self.max_work_group_size = device.max_work_group_size
        else:
            # these settings are for testing only
            self.max_work_group_size = 512
            self.max_local_group_sizes = [512, 512, 512]
            self.max_compute_units = 40


    def compute_local_size_1d(self, shape):
        """
        1d local_size
        :param shape:
        :return:
        """
        return min(shape[0]/2, self.max_local_group_sizes[0])

    def compute_local_size_2d(self, shape):
        """
        2d local size
        :param shape:
        :return:
        """
        d0_size, d1_size = 1, 1
        grid_size = product(shape)
        if grid_size < self.max_work_group_size:
            return tuple(shape)

        def get_work_group_for_divisor(dim_divisor):
            adjust = 0 if shape[1] % 2 == 0 else 1
            d1_size = min((shape[1]/dim_divisor) + adjust, self.max_local_group_sizes[1])
            d0_size = min(self.max_work_group_size / d1_size, self.max_local_group_sizes[0])
            return d0_size, d1_size

        def compute_error(work_group_size):
            dimensions = len(work_group_size)

            work_groups_per_dim = [
                ((shape[n]-1)/work_group_size[n])+1
                for n in range(dimensions)
            ]
            total_work_groups = product(work_groups_per_dim)

            def error_for_dim(dim):
                remainder = shape[dim] % work_group[dim]
                if remainder == 0:
                    return 0.0
                else:
                    dimension_weight = work_groups_per_dim[dim]/float(total_work_groups)
                    return (work_group[dim] - remainder) * dimension_weight

            error = sum([
                error_for_dim(n)
                for n in range(dimensions)
            ])
            return error

        best_work_group = None
        minimum_error = None
        for divisor in range(1, 8):
            work_group = get_work_group_for_divisor(divisor)
            error = compute_error(work_group)

            print("shape {} work_group {} error {}".format(shape, work_group, error))

            if error == 0.0:
                return work_group

            if minimum_error is None or minimum_error > error:
                minimum_error = error
                best_work_group = work_group

        return best_work_group


