from __future__ import print_function
__author__ = 'chick'

import math
import pycl

import itertools


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

    def compute_local_size_thin(self, shape):
        """
        compute a local size that leans toward maximizing the length
        along the rightmost index of shape.
        in that domain, try and minimize the overshoot when the local
        size cannot be an exact multiple of the global_size
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

            # print("shape {} work_group {} error {}".format(shape, work_group, error))

            if error == 0.0:
                return work_group

            if minimum_error is None or minimum_error > error:
                minimum_error = error
                best_work_group = work_group

        return best_work_group

    def compute_local_size_lenny_style(self, shape):
        max_sizes = self.max_local_group_sizes
        max_total = self.max_work_group_size

        if sum([x % 2 for x in shape]) == len(shape):
            return tuple([1 for _ in shape])
        if len(shape) == 3:
            x_len, y_len, z_len = 1, 1, 1
            last_tuple = (0, 0, 0)
            while (x_len, y_len, z_len) != last_tuple:
                if shape[0] % 2 == 1:
                    x_len = 1
                else:
                    x_len = min(max_sizes[0], x_len * 2)
                if max_total - z_len * x_len * y_len <= 0:
                    break
                if shape[1] % 2 == 1:
                    y_len = 1
                else:
                    y_len = min(max_sizes[1], y_len * 2)
                if max_total - z_len * x_len * y_len <= 0:
                    break
                if shape[2] % 2 == 1:
                    z_len = 1
                else:
                    z_len = min(max_sizes[2], z_len * 2)
                if max_total - z_len * x_len * y_len <= 0:
                    break
                if x_len == shape[0] or \
                        y_len == shape[1] or \
                        z_len == shape[2]:
                    break
                last_tuple = (x_len, y_len, z_len)

            local_size = (x_len, y_len, z_len)
        elif len(shape) == 2:
            x_len, y_len = 1, 1
            last_tuple = (0, 0)
            while (x_len, y_len) != last_tuple:
                if shape[0] % 2 == 1:
                    x_len = 1
                else:
                    x_len = min(max_sizes[0], x_len * 2)
                if max_total - x_len * y_len <= 0:
                    break
                if shape[1] % 2 == 1:
                    y_len = 1
                else:
                    y_len = min(max_sizes[1], y_len * 2)
                if max_total - x_len * y_len <= 0:
                    break
                if x_len == shape[0] or \
                        y_len == shape[1]:
                    break
                last_tuple = (x_len, y_len)

            local_size = (x_len, y_len)
        else:
            local_size = (min(
                max_total, max_sizes[0], shape[0] / 2),)

        return local_size

    def get_a_bulky_range(self, dims_remaining, cur_max_size, max_local):
        """
        return a reasonable range of sizes to try for
        :param cur_shape:
        :param cur_max_size:
        :return:
        """
        target_size = int((cur_max_size ** (1.0 / dims_remaining)) + 0.5)

        for size in range(max(2, target_size-10), min(max_local, target_size+10)):
            yield size

    def get_local_size(self, shape, dim, max_size, local_size=None):
        if local_size is None:
            local_size = []
        if dim >= len(shape)-1:
            new_local_size = local_size + [max_size]
            yield tuple(new_local_size)
        else:
            for size in self.get_a_bulky_range(len(shape)-dim, max_size, self.max_local_group_sizes[dim]):
                new_local_size = local_size + [size]
                for x in self.get_local_size(
                        shape, dim+1, max_size // size, new_local_size):
                    if product(x) > 0:
                        yield x

    def dimension_processing_priority_key(self, shape):
        """
        return a key that indicates the priority for processing a particular dimensions
        dimensions that have more constraints should be processed before
        :param shape:
        :return:
        """
        pass

    def compute_local_size_bulky(self, shape):
        """
        compute a local size that leans toward minimizing the surface area to volume
        ratio of the n-dimensional local_size shape.
        in that domain, try and minimize the overshoot when the local
        size cannot be an exact multiple of the global_size
        :param shape:
        :return:
        """
        best_local_size = None
        largest_volume = 0
        for candidate_local_size in self.get_local_size(shape, 0, self.max_work_group_size):
            ratio = product(candidate_local_size) / (2.0 * sum(candidate_local_size))
            # print("shape {:12} local_size {:12} product {:12} sum {:12} ratio {:12}".format(
            #     shape, candidate_local_size,
            #     product(candidate_local_size), 2 * sum(candidate_local_size), ratio
            # ))
            if ratio > largest_volume:
                largest_volume = ratio
                best_local_size = candidate_local_size

        best_local_size = [min(shape[dim], value) for dim, value in enumerate(best_local_size)]
        return tuple(best_local_size)

    def compute_local_size(self, shape, method=None):
        if method is None or method == 'thin':
            return self.compute_local_size_thin(shape)
        else:
            return self.compute_local_bulky(shape)


class LocalSizeComputer(object):
    def __init__(self, shape, device=None, even_multiples_only=True):
        self.shape = shape[:]
        self.dimensions = len(shape)
        self.even_multiples_only = even_multiples_only
        if device is None:
            try:
                device = pycl.clGetDeviceIDs()[-1]
                self.max_local_group_sizes = pycl.clGetDeviceInfo(
                    device, pycl.cl_device_info.CL_DEVICE_MAX_WORK_ITEM_SIZES)
                self.max_work_group_size = pycl.clGetDeviceInfo(
                    device, pycl.cl_device_info.CL_DEVICE_MAX_WORK_GROUP_SIZE)
                self.compute_units = pycl.clGetDeviceInfo(
                    device, pycl.cl_device_info.CL_DEVICE_MAX_COMPUTE_UNITS)
            except:
                self.max_work_group_size = 512
                self.max_local_group_sizes = [512, 512, 512]
                self.compute_units = 40
        else:
            self.max_local_group_sizes = device.max_local_group_sizes
            self.max_work_group_size = device.max_work_group_size
            self.compute_units = device.max_compute_units

        overshoot = 1.5
        #
        # make a first estimate of the largest index to consider in each dimension
        # that will be the n-th root of the max work group size in order to minimize surface area to volume ratio
        self.root_size = int((self.max_work_group_size ** (1.0 / self.dimensions)) + 0.5)
        self.max_indices = [
            int(self.root_size * overshoot)
            for dim in range(self.dimensions)
        ]
        #
        # adjust each dimension downward if it exceeds the max_local_size for that dimension
        # adjust the other dimensions upward if there is room
        for dim in range(self.dimensions):
            if self.max_indices[dim] > self.max_local_group_sizes[dim]:
                self.max_indices[dim] = self.max_local_group_sizes[dim]
                indices_to_fix = []
                for dim2 in range(self.dimensions):
                    if dim2 != dim and self.max_indices[dim2] < self.max_local_group_sizes[dim2]:
                        indices_to_fix.append(dim2)
                if len(indices_to_fix) > 0:
                    new_root = int(int(self.max_work_group_size ** (1.0 / len(indices_to_fix)) + 0.5) * 1.5)
                    for dim2 in indices_to_fix:
                        self.max_indices[dim2] = min(new_root, self.max_local_group_sizes[dim2])

        # if the indices we have selected so far are significantly larger than the size of the target matrix
        # then adjust them that dimensions index downward and adjust upward any trailing indices
        for dim in range(self.dimensions):
            if self.shape[dim] * overshoot < self.max_indices[dim]: #  and self.shape[dim] < self.max_local_group_sizes[dim]:
                self.max_indices[dim] = min(self.max_local_group_sizes[dim], int(self.shape[dim] * overshoot))
                if dim == 0:  # increase the size of the other dimensions
                    if self.dimensions == 2:
                        self.max_indices[1] = max(
                            int((self.max_work_group_size / self.max_indices[0]) * overshoot),
                            self.max_local_group_sizes[1]
                        )
                    if self.dimensions == 3:
                        temp_root = int(math.sqrt(self.max_work_group_size / self.max_indices[0]) * overshoot)
                        self.max_indices[1] = min(temp_root, self.max_local_group_sizes[1])
                        self.max_indices[2] = min(temp_root, self.max_local_group_sizes[2])
                elif dim == 1:  # increase the size of the remaining direction
                    if self.dimensions == 3:
                        self.max_indices[2] = max(
                            int((self.max_work_group_size / (self.max_indices[0] * self.max_indices[1])) * overshoot),
                            self.max_local_group_sizes[2]
                        )

    def get_local_size(self, dim, max_size, local_size=None):
        if local_size is None:
            local_size = []
        if dim >= self.dimensions-1:
            new_local_size = local_size + [min(max_size, self.max_local_group_sizes[dim])]
            yield tuple(new_local_size)
        else:
            for size in range(max(1, self.max_indices[dim] - 20), self.max_indices[dim]+1):
                new_local_size = local_size + [size]
                for x in self.get_local_size(
                        dim+1, max_size // size, new_local_size):
                    if product(x) > 0:
                        yield x

    def dimension_processing_priority_key(self, dimension):
        """
        return a key that indicates the priority for processing a particular dimensions
        dimensions that have more constraints should be processed before
        :param shape:
        :return:
        """
        if self.shape[dimension] > self.max_local_group_sizes[dimension]:
            return self.shape[dimension] + dimension
        if self.shape[dimension] < 10 * self.max_local_group_sizes[dimension]:
            return int(self.shape[dimension] + 1e6)
        return int(self.shape[dimension] / self.max_local_group_sizes[dimension] + 2e6)

    def get_ordered_dims(self):
        def get_key(dim):
            return self.dimension_processing_priority_key(dim)
        dims = [d for d in range(self.dimensions)]
        return sorted(dims, key=get_key)

    @staticmethod
    def volume(vector):
        if len(vector) == 1:
            return vector[0] ** 2
        return product(vector)

    @staticmethod
    def surface_area(vector):
        if len(vector) == 1:
            return vector[0]
        elif len(vector) == 2:
            return sum(vector) * 2
        return sum([product([f * 2 for f in face]) for face in itertools.permutations(vector, len(vector)-1)])

    def compute_local_size_bulky(self):
        """
        compute a local size that leans toward minimizing the surface area to volume
        ratio of the n-dimensional local_size shape.
        in that domain, try and minimize the overshoot when the local
        size cannot be an exact multiple of the global_size
        :param shape:
        :return:
        """
        best_local_size = None
        largest_volume = 0
        for candidate_local_size in self.get_local_size(0, self.max_work_group_size):
            ratio = ( LocalSizeComputer.volume(candidate_local_size)) / \
                float(LocalSizeComputer.surface_area(candidate_local_size))
            # print("shape {:12} local_size {:12} product {:12} sum {:12} ratio {:12}".format(
            #     self.shape, candidate_local_size,
            #     LocalSizeComputer.volume(candidate_local_size),
            #     LocalSizeComputer.surface_area(candidate_local_size),
            #     ratio
            # ))
            if ratio > largest_volume:
                largest_volume = ratio
                best_local_size = candidate_local_size

        best_local_size = [min(self.shape[dim], value) for dim, value in enumerate(best_local_size)]
        return tuple(best_local_size)

    def get_work_group_for_divisor(self, dim_divisor):
        """
        generated a legal work group size when dividing up the right most dimension
        in dim_divisor pieces,
        the adjustment tries to make the division of the shape a little bigger to so the 1/dim_divisor
        sizes will just cover the shape in that dimension
        :param dim_divisor:
        :return: a tuple of the same cardinality as self.shape
        """
        last_dim = self.dimensions - 1
        penultimate_dim = last_dim - 1
        adjust = 0 if self.shape[last_dim] % 2 == 0 or dim_divisor == 1 else 1
        last_dim_size = max(1, min(
            int((self.shape[last_dim]/dim_divisor) + adjust),
            self.max_local_group_sizes[last_dim]
        ))
        penultimate_dim_size = min(
            int(self.max_work_group_size / last_dim_size),
            self.max_local_group_sizes[penultimate_dim], self.shape[penultimate_dim]
        )
        if self.dimensions == 2:
            return penultimate_dim_size, last_dim_size

        first_dim_size = min(
            int(self.max_work_group_size / (last_dim_size * penultimate_dim_size)),
            self.max_local_group_sizes[0], self.shape[0]
        )
        return first_dim_size, penultimate_dim_size, last_dim_size

    def compute_error(self, work_group_size):
        dimensions = len(work_group_size)

        work_groups_per_dim = [
            int((self.shape[n]-1)/work_group_size[n])+1
            for n in range(dimensions)
        ]
        total_work_groups = product(work_groups_per_dim)

        def error_for_dim(dim):
            remainder = self.shape[dim] % work_group_size[dim]
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

    def compute_local_size_thin(self):
        """
        compute a local size that leans toward maximizing the length
        along the rightmost index of shape.
        in that domain, try and minimize the overshoot when the local
        size cannot be an exact multiple of the global_size
        :return:
        """
        if self.dimensions == 1:
            return (max(1, min(int(self.shape[0]/2), self.max_local_group_sizes[0])),)

        best_work_group = None
        minimum_error = None
        for divisor in range(1, 8):
            work_group = self.get_work_group_for_divisor(divisor)
            error = self.compute_error(work_group)

            # print("self.shape {} work_group {} error {}".format(self.shape, work_group, error))

            if error == 0.0:
                return work_group

            if minimum_error is None or minimum_error > error:
                minimum_error = error
                best_work_group = work_group

        return best_work_group
