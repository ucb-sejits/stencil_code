import numpy
import sys

__author__ = 'chick'


class Neighborhood(object):
    """
    convenience class for defining and testing neighborhoods

    """
    @staticmethod
    def origin_of_dim(dim):
        return tuple([0 for _ in range(dim)])

    @staticmethod
    def flatten(l):
        """
        from http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
        :param l:
        :return:
        """
        list_type = type(l)
        l = list(l)
        i = 0
        while i < len(l):
            while isinstance(l[i], list):
                if not l[i]:
                    l.pop(i)
                    i -= 1
                    break
                else:
                    l[i:i + 1] = l[i]
            i += 1
        return list_type(l)

    @staticmethod
    def neighborhood_generator(r, d, l=list()):
        if d == 0:
            return tuple(l)
        return [Neighborhood.neighborhood_generator(r, d-1, l + [x]) for x in range(-r, r+1)]

    @staticmethod
    def von_neuman_neighborhood(radius=1, dim=2, include_origin=True):
        """
        create a neighborhood of points around origin where
        taxi distance is less than or equal to radius
        """
        def within_legal_taxi_distance(p):
            return sum([abs(x) for x in list(p)]) <= radius

        points = filter(within_legal_taxi_distance, Neighborhood.flatten(Neighborhood.neighborhood_generator(radius, dim)))
        if not include_origin:
            points.remove(Neighborhood.origin_of_dim(dim))
        return points

    @staticmethod
    def moore_neighborhood(radius=1, dim=2, include_origin=True):
        """
        create a neighborhood of points around origin where each coordinate is
        less than or equal to r
        """
        points = Neighborhood.flatten(Neighborhood.neighborhood_generator(radius, dim))
        if not include_origin:
            points.remove(Neighborhood.origin_of_dim(dim))
        return points

    @staticmethod
    def compute_halo(point_list):
        """

        :param point_list: list of point tuples, all of them better be same dimension
        :param shape: a numpy style shape specification
        :return: halo dimension tuple of tuple(negative_axis_magnitude,positive_axis_magnitude)
        """
        halo = None
        if len(point_list) > 0:
            dim = len(point_list[0])
            halo = numpy.array([(0, 0) for x in range(dim)])

            for point in point_list:
                for d in range(dim):
                    if halo[d][0] < abs(point[d]):
                        halo[d][0] = abs(point[d])
                    if halo[d][1] < abs(point[d]):
                        halo[d][1] = abs(point[d])
        return tuple(halo)

    @staticmethod
    def compute_from_indices(matrix, mid_point=None):
        """
        finds a neighborhood containing the indices of matrix where the value is not zero
        also compute the halo min and max for for each dimension
        :param matrix: an n-dimensional matrix of coefficients
        :param mid_point: if defined, it represents the origin point of the matrix, if
        not defined it is set to be the halfway point along each dimension of matrix
        :return: (neighbor_point_list, coefficient_list, halo_min_and_max_vector)
        neighbor_point_list and coefficient_list are ordered by the same index
        """
        matrix = numpy.array(matrix)  # we wrap to get shape etc
        neighbor_points = []
        coefficients = []
        it = numpy.nditer(matrix, flags=['multi_index'])
        dim = len(matrix.shape)
        if mid_point is None:
            mid_point = [x / 2 for x in matrix.shape]
        while not it.finished:
            value = it[0]
            index = it.multi_index
            if value != 0:
                relative_point = tuple([it.multi_index[d]-mid_point[d] for d in range(dim)])
                neighbor_points.append(relative_point)
                coefficients.append(matrix[index])
            it.iternext()

        halo = Neighborhood.compute_halo(neighbor_points)

        return neighbor_points, coefficients, halo




