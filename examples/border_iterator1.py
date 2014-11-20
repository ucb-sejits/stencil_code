__author__ = 'chick'
import copy


class HaloEnumerator(object):
    """
    Enumerates all points of n-dimensional matrix described by shape that are on the exterior of the matrix
    where the exterior thickness is described by halo, halo must be the same size as shape.
    Currently halos have only one value for each dimension though it perhaps should be built to accommodate
    a separate value for low indices and high indices
    """
    def __init__(self, halo, shape):
        """
        :param halo: a enumerable list of the size of the halo in each dimension
        :param shape: the size of each dimension of the matrix to iterate over
        :return:
        """
        self.halo = halo
        self.shape = shape

    def other_dimension_iterator(self, fixed_dimension, dimension, constraint, point):
        if dimension == fixed_dimension:
            dimension += 1
        if dimension < len(self.shape):
            # print("{}odi dimension {} constraint {}".format(" "*dimension, dimension, constraint))
            for dimension_index in range(constraint[dimension][0], constraint[dimension][1]):
                point[dimension] = dimension_index
                for border_point in self.other_dimension_iterator(
                        fixed_dimension, dimension+1, constraint, copy.deepcopy(point)):
                    yield border_point
        else:
            yield tuple(point)

    def fixed_surface_iterator(self, dimension, constraint):
        """
        begins the recursive construction of a point, this function
        iterates over a series of n-planes on the low side of a dimension then
        the corresponding n-planes on the high side.  Updating of the constraint
        avoids point revisiting
        :param dimension:
        :param constraint:
        :return:
        """
        for fixed_surface_index in range(0, self.halo[dimension]):
            point = [0 for _ in self.shape]
            point[dimension] = fixed_surface_index
            for surface_point in self.other_dimension_iterator(dimension, 0, constraint, point):
                yield surface_point

        for fixed_surface_index in range(self.shape[dimension]-self.halo[dimension], self.shape[dimension]):
            point = [0 for _ in self.shape]
            point[dimension] = fixed_surface_index
            for surface_point in self.other_dimension_iterator(dimension, 0, constraint, point):
                yield surface_point

        constraint[dimension][0] = self.halo[dimension]
        constraint[dimension][1] = self.shape[dimension] - self.halo[dimension]

    def __iter__(self):
        """
        This top level iterator will iterate over successive n-planes at either end of each dimension
        After each dimension is processed the constraint vector is updated for that dimension so points
        are not revisited
        :return:
        """
        constraint = [[0, size] for size in self.shape]

        for dimension in range(len(self.shape)):
            # print("constraint {}".format(constraint))
            for next_point in self.fixed_surface_iterator(dimension, constraint):
                yield next_point


for border_point in HaloEnumerator([1, 1], [5, 5]):
    print("border point {}".format(border_point))
