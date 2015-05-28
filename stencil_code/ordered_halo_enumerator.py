from __future__ import print_function
import itertools

__author__ = 'chick'


class OrderedHaloEnumerator(object):
    """
    Iterates over the surface of a n-dimensional grid where the surface has a depth described by the halo vector
    It uses the notion of a surface key defined as a n-tuple whose each element is an i to indicate the surface
    is on the interior (i) or edge (e) for that dimension.  the iterations are ordered such that surfaces that
    are only on an edge in one dimension (i.e. the surface key has only one 'e' in it) are done first, then
    the surfaces keys with 2 'e's and so forth ending with corners, surface keys consisting of all 'e's
    """
    def __init__(self, halo, shape):
        """
        :param halo: a enumerable list of the size of the halo in each dimension
        :param shape: the size of each dimension of the matrix to iterate over
        :return:
        """
        assert len(halo) == len(shape), \
            "OrderedHaloEnumerator halo({}) and shape({}) must be same size".format(halo, shape)

        self.halo = halo
        self.shape = shape
        self.dimensions = len(self.shape)

    def surface_iterator(self, surface_key, dim=0):
        if surface_key[dim] == 'i':
            start, stop = self.halo[dim], self.shape[dim] - self.halo[dim]
            for index in range(start, stop):
                if dim < self.dimensions-1:
                    for lower_indices in self.surface_iterator(surface_key, dim+1):
                        yield (index,) + lower_indices
                else:
                    yield (index,)
        else:
            start, stop = 0, self.halo[dim]

            for index in range(start, stop):
                if dim < self.dimensions-1:
                    for lower_indices in self.surface_iterator(surface_key, dim+1):
                        yield (index,) + lower_indices
                else:
                    yield (index,)

            start, stop = self.shape[dim]-self.halo[dim], self.shape[dim]

            for index in range(start, stop):
                if dim < self.dimensions-1:
                    for lower_indices in self.surface_iterator(surface_key, dim+1):
                        yield (index,) + lower_indices
                else:
                    yield (index,)

    def ordered_border_type_enumerator(self):
        """
        border types are faces, edges, corners, hyper-corners, ...
        :return:
        """
        def num_edges(vec):
            return len(list(filter(lambda x: x == 'e', vec)))
        return sorted(
            list(itertools.product('ie', repeat=self.dimensions))[1:],
            key=num_edges
        )

    def __iter__(self):
        """
        This top level iterator will iterate over each class of border surface type
        ending with the corners
        :return:
        """
        for border_key in self.ordered_border_type_enumerator():
            for border_point in self.surface_iterator(border_key):
                yield border_point

if __name__ == '__main__':  # pragma no cover
    ordered_halo_enumerator = OrderedHaloEnumerator([1, 1], [3, 3])
    for halo_key in ordered_halo_enumerator.ordered_border_type_enumerator():
        for halo_point in ordered_halo_enumerator.surface_iterator(halo_key):
            print("{} point {} ".format(halo_key, halo_point))
    # for halo_index in OrderedHaloEnumerator([1, 1], [5, 5]):
    #     print("border point {}".format(halo_index))
