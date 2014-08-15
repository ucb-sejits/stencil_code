import numpy
from itertools import product, ifilterfalse


class StencilGrid(object):
    """
    A two-dimension grid of numeric values, used for input and output to a
    stencil kernel.
    """

    def __init__(self, shape, dtype=numpy.float32, data=None, neighbors=None):
        """__init__

        :param shape: the shape of the StencilGrid, e.g. `(1024, 1024)`
        :type shape: int or sequence of ints
        """
        self.dim = len(shape)
        if data is not None:
            self.data = data
        else:
            self.data = numpy.zeros(shape, dtype=dtype)

        self.shape = shape
        self.ghost_depth = 1
        self.grid_variables = []
        self.interior = []

        self.set_grid_variables()
        self.set_interior()
        # add default neighbor definition
        if neighbors is not None:
            # TODO: Check validity of neighbor defn
            self.neighbor_definition = neighbors
        else:
            self.set_default_neighbor_definitions()
        self.corner_points = None
        self.edge_points = None
        self.make_corner_points_iterator()
        self.make_edge_points_iterator()

        # import types
        #
        # code = ["def corner_points(self):"]
        # for dimension_index in range(self.dim):
        #     for each_dimension in range(self.dim):
        #         if each_dimension == dimension_index:
        #             code.append(
        #                 "%sfor d%s in [0,%d]:" %
        #                 (' '*4*(each_dimension+1), each_dimension, self.shape[each_dimension])
        #             )
        #         else:
        #             code.append(
        #                 "%sfor d%s in range(%d):" %
        #                 (' '*4*(each_dimension+1), each_dimension, self.shape[each_dimension])
        #             )
        #     code.append("%syield (%s)" % (' '*4*(self.dim+1), ",".join(map(lambda x: "d%d" % x, range(self.dim)))))
        # for line in code:
        #     print(line)
        # exec('\n'.join(code))
        # self.corner_points = types.MethodType(corner_points, self)

    # want this to be indexable
    def __getitem__(self, x):
        """__getitem__

        :param x: The index of the StencilGrid to return. Equivalent to indexing
                  a numpy array.
        :type x: int
        """
        return self.data[x]

    def __setitem__(self, x, y):
        """__setitem__

        :param x: The index of the StencilGrid to set. Equivalent to setting a
                  numpy array.
        :type x: int
        :param y: The value to set at index `x`.
        :type y: `self.data.dtype`
        """
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
        """
        create a neighborhood of all adjacent points along
        coordinate axes, suitable for the dimension of this instance
        """
        neighborhood = []
        origin = [0 for _ in range(self.dim)]
        for dimension in range(self.dim):
            for offset in [-1, 1]:
                point = origin[:]
                point[dimension] = offset
                neighborhood.append(tuple(point))

        return neighborhood

    def moore_neighborhood(self, include_origin=False):
        """
        create a neighborhood of all adjacent points along
        coordinate axes
        """

        neighborhood_list = []

        def dimension_iterator(dimension, point_accumulator):
            """
            accumulates into local neighborhood_list
            """
            if dimension >= self.dim:
                if include_origin or sum([abs(x) for x in point_accumulator]) != 0:
                    neighborhood_list.append(tuple(point_accumulator))
            else:
                for dimension_coordinate in [-1, 0, 1]:
                    new_point_accumulator = point_accumulator[:]
                    new_point_accumulator.append(dimension_coordinate)
                    dimension_iterator(
                        dimension+1,
                        new_point_accumulator
                    )

        dimension_iterator(0, [])

        return neighborhood_list

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
        all_dims = [range(self.ghost_depth, self.shape[x]-self.ghost_depth) for x in range(0, self.dim)]
        for item in itertools.product(*all_dims):
            yield tuple(item)

    def make_edge_points_iterator(self):
        """
        creates an iterator for edge points of the stencil.  This is done by dynamically compiling code
        because it is difficult to create an iterator for the nesting of for loops over and arbitrary
        shape of the grid.  Edge points are defined to be for each dimension the scalar values of the
        corner points in that dimension combined with the iteration of all scalars from the other dimensions
        omitting their corner values
        """
        import types

        edge_points = None
        code = [
            "def edge_points(self):",
            "    seen = set()",
            "    rejected = 0",
        ]
        for dimension_index in range(self.dim):
            for each_dimension in range(self.dim):
                if each_dimension == dimension_index:
                    border_points = range(self.ghost_depth) + \
                        range(self.shape[each_dimension]-self.ghost_depth, self.shape[each_dimension])
                    code.append("%sfor d%s in [%s]:" %
                                (
                                    ' '*4*(each_dimension+1),
                                    each_dimension,
                                    ','.join(map(lambda x: "%d" % x, border_points))
                                )
                                )
                elif (each_dimension - 1 + self.dim) % self.dim == dimension_index:
                    border_points = range(self.ghost_depth, self.shape[each_dimension]-self.ghost_depth)
                    code.append("%sfor d%s in [%s]:" %
                                (
                                    ' '*4*(each_dimension+1),
                                    each_dimension,
                                    ','.join(map(lambda x: "%d" % x, border_points))
                                )
                                )
                else:
                    code.append("%sfor d%s in range(%s):" %
                                (
                                    ' '*4*(each_dimension+1),
                                    each_dimension,
                                    self.shape[each_dimension]
                                )
                                )
            code.append("%spoint = (%s)" % (' '*4*(self.dim+1), ",".join(map(lambda x: "d%d" % x, range(self.dim)))))
            code.append("%sif not seen.__contains__(point):" % (' '*4*(self.dim+1)))
            code.append("%sseen.add(point)" % (' '*4*(self.dim+2)))
            code.append("%syield point" % (' '*4*(self.dim+2)))
            # uncomment out below to see how many points rejected as already seen
            # code.append("%selse:" % (' '*4*(self.dim+1)))
            # code.append("%srejected += 1" % (' '*4*(self.dim+2)))
            # code.append('    print "rejected %d points" % rejected' )

        # uncomment to see generated code
        # for line in code:
        #     print(line)
        exec('\n'.join(code))
        self.edge_points = types.MethodType(edge_points, self)

    def make_corner_points_iterator(self):
        """
        creates an iterator for border points of the stencil.  This is done by dynamically compiling code
        because it is difficult to create an iterator for the nesting of for loops over and arbitrary
        shape of the grid.  Border points are defined to be iteration over all points within ghost_depth
        of the min and max values of each dimension
        """
        import types

        corner_points = None
        code = ["def corner_points(self):"]
        for each_dimension in range(self.dim):
            border_points = range(self.ghost_depth) + \
                range(self.shape[each_dimension]-self.ghost_depth, self.shape[each_dimension])
            code.append("%sfor d%s in [%s]:" %
                        (
                            ' '*4*(each_dimension+1),
                            each_dimension,
                            ','.join(map(lambda x: "%d" % x, border_points))
                        )
                        )
        code.append("%syield (%s)" % (' '*4*(self.dim+1), ",".join(map(lambda x: "d%d" % x, range(self.dim)))))
        # for line in code:
        #     print(line)
        exec('\n'.join(code))
        self.corner_points = types.MethodType(corner_points, self)

    def border_points(self):
        """
        Iterator over the border points of a grid.  Only executed in pure Python
        mode; in SEJITS mode, it should be executed only in the translated
        language/library.

        Border points are the sequential iteration of all corner points
        followed by all edge points
        Note: boundary points is slightly faster but
        """

        for point in self.corner_points():
            yield point
        for point in self.edge_points():
            yield point

    def boundary_points(self):
        """
        different technique using itertools to compute boundary points of a grid
        This method does not work if ghost_depth is != 1
        """
        assert self.ghost_depth == 1, "ghost depth not 1, use border points instead"
        dims = map(lambda x: (0, x-1), self.shape)
        seen = set()
        # rejected = 0
        ranges = [xrange(lb, ub+1) for lb, ub in dims]
        for i, dim in enumerate(dims):
            for bound in dim:
                spec = ranges[:i] + [[bound]] + ranges[i+1:]
                for pt in ifilterfalse(seen.__contains__, product(*spec)):
                    seen.add(pt)
                    yield pt
                # commented out code equivalent to above but tracks dups
                # for pt in product(*spec):
                #     if not seen.__contains__(pt):
                #         seen.add(pt)
                #         yield pt
                #     else:
                #         rejected += 1
        # print "boundary points rejected %d" % rejected

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
            yield tuple(map(lambda a, b: a+b, list(center), list(neighbor)))

    def __repr__(self):
        return self.data.__repr__()
