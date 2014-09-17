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
