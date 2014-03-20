"""
A two-dimension grid of numeric values, used for input and output to a stencil kernel.
"""

import numpy


class StencilGrid(object):

    def __init__(self, size):
        self.dim = len(size)
        self.data = numpy.zeros(size)
        self.shape = size
        self.ghost_depth = 1
        self.grid_variables = []
        self.interior = []

        self.set_grid_variables()
        self.set_interior()
        # add default neighbor definition
        self.neighbor_definition = []
        self.set_default_neighbor_definitions()

    # want this to be indexable
    def __getitem__(self, x):
        return self.data[x]

    def __setitem__(self, x, y):
        self.data[x] = y

    def set_grid_variables(self):
        self.grid_variables = ["DIM"+str(x) for x in range(0, self.dim)]

    def set_interior(self):
        """
        Sets the number of interior points in each dimension
        """
        self.interior = [x-2*self.ghost_depth for x in self.shape]

    def set_neighborhood(self, neighborhood_id, coordinate_list):
        """
        a grid can one or more notions of a neighborhood of a given grid point
        neighborhood_id is an integer identifier of the neighborhood
        coordinate_list is a list of tuples appropriate to the shape of the grid
        """
        for coordinate in coordinate_list:
            assert len(coordinate) == self.dim, "neighborhood coordinates must be of proper dimension"

        while len(self.neighbor_definition) <= neighborhood_id:
            self.neighbor_definition.append([])
        self.neighbor_definition[neighborhood_id] = coordinate_list

    def von_neuman_neighborhood(self):
        neighborhood = []
        origin = [0 for _ in range(self.dim)]
        for dimension in range(self.dim):
            for offset in [-1, 1]:
                point = origin[:]
                point[dimension] = offset
                neighborhood.append(tuple(point))

        return neighborhood

    def set_default_neighbor_definitions(self):
        """
        Sets the default for neighbors[0] and neighbors[1].  Note that neighbors[1]
        does not include the center point.
        """
        self.neighbor_definition = []

        zero_point = tuple([0 for _ in range(self.dim)])
        self.neighbor_definition.append([zero_point])
        self.neighbor_definition.append(self.von_neuman_neighborhood())

    def interior_points(self):
        """
        Iterator over the interior points of the grid.  Only executed
        in pure Python mode; in SEJITS mode, it should be executed only
        in the translated language/library.
        """
        import itertools
        all_dims = [range(self.ghost_depth,self.shape[x]-self.ghost_depth) for x in range(0,self.dim)]
        for item in itertools.product(*all_dims):
            yield tuple(item)

    def border_points(self):
        """
        Iterator over the border points of a grid.  Only executed in pure Python
        mode; in SEJITS mode, it should be executed only in the translated
        language/library.
        """
        # TODO
        return []

    def neighbors(self, center, neighbors_id):
        """
        Returns the list of neighbors with the given neighbors_id. By
        default, IDs 0 and 1 give the list consisting of all
        points at a distance of 0 and 1 from the center point,
        respectively. Uses neighbor_definition to determine what the
        neighbors are.
        """
        # import pprint
        # print( "neighbors_id %s" % neighbors_id )
        # pprint.pprint(self.neighbor_definition)
        # return tuples for each neighbor
        for neighbor in self.neighbor_definition[neighbors_id]:
            yield tuple(map(lambda a,b: a+b, list(center), list(neighbor)))

    def __repr__(self):
        return self.data.__repr__()
