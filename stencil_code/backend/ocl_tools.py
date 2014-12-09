from __future__ import print_function
__author__ = 'chick'


def product(vector):
    result = 1
    for element in vector:
        result *= element
    return result


class OclTools(object):
    def __init__(self, device=None):
        if device is not None:
            self.max_work_group_size = device.max_work_group_size
            self.max_local_group_sizes = device.max_local_group_sizes
            self.max_compute_units = device.max_compute_units
        else:
            # these settings are for testing only
            self.max_work_group_size = 512
            self.max_local_group_sizes = [512, 512, 512]
            self.max_compute_units = 40

    def get_work_group_for_divisor(self, shape, dim_divisor):
        """
        generated a legal work group size when dividing up the right most dimension
        in dim_divisor pieces,
        the adjustment tries to make the division of the shape a little bigger to so the 1/dim_divisor
        sizes will just cover the shape in that dimension
        :param dim_divisor:
        :return: a tuple of the same cardinality as shape
        """
        last_dim = len(shape) - 1
        penultimate_dim = last_dim - 1
        adjust = 0 if shape[last_dim] % 2 == 0 or dim_divisor == 1 else 1
        last_dim_size = max(1, min(
            int((shape[last_dim]/dim_divisor) + adjust),
            self.max_local_group_sizes[last_dim]
        ))
        penultimate_dim_size = min(
            int(self.max_work_group_size / last_dim_size),
            self.max_local_group_sizes[penultimate_dim], shape[penultimate_dim]
        )
        if len(shape) == 2:
            return penultimate_dim_size, last_dim_size

        first_dim_size = min(
            int(self.max_work_group_size / (last_dim_size * penultimate_dim_size)),
            self.max_local_group_sizes[0], shape[0]
        )
        return first_dim_size, penultimate_dim_size, last_dim_size

    def compute_error(self, shape, work_group_size):
        dimensions = len(work_group_size)

        work_groups_per_dim = [
            int((shape[n]-1)/work_group_size[n])+1
            for n in range(dimensions)
        ]
        total_work_groups = product(work_groups_per_dim)

        def error_for_dim(dim):
            remainder = shape[dim] % work_group_size[dim]
            if remainder == 0:
                return 0.0
            else:
                dimension_weight = work_groups_per_dim[dim]/float(total_work_groups)
                return (work_group_size[dim] - remainder) * dimension_weight

        local_error = sum([
            error_for_dim(n)
            for n in range(dimensions)
        ])
        return local_error

    def compute_local_size(self, shape):
        """
        3d local size
        :param shape:
        :return:
        """
        if len(shape) == 1:
            return (max(1, min(int(shape[0]/2), self.max_local_group_sizes[0])),)

        best_work_group = None
        minimum_error = None
        for divisor in range(1, 8):
            work_group = self.get_work_group_for_divisor(shape, divisor)
            error = self.compute_error(shape, work_group)

            print("shape {} work_group {} error {}".format(shape, work_group, error))

            if error == 0.0:
                return work_group

            if minimum_error is None or minimum_error > error:
                minimum_error = error
                best_work_group = work_group

        return best_work_group


