__author__ = 'chick'
import copy

shape = [5, 5]
halo = [1, 1]


def other_dimension_iterator(fixed_dimension, dimension, constraint, halo, shape, point):
    if dimension == fixed_dimension:
        other_dimension_iterator(fixed_dimension, dimension+1, constraint, halo, shape, copy.deepcopy(point))
    elif dimension < len(shape):
        print("{}odi dimension {} constraint {}".format(" "*dimension, dimension, constraint))
        for dimension_index in range(constraint[dimension][0], constraint[dimension][1]):
            point[dimension] = dimension_index
            other_dimension_iterator(fixed_dimension, dimension+1, constraint, halo, shape, copy.deepcopy(point))
    else:
        print("point {}".format(tuple(point)))

def fixed_surface_iterator(dimension, constraint, halo, shape):
    for fixed_surface_index in range(0, halo[dimension]) + range(shape[dimension]-halo[dimension], shape[dimension]):
        point = [0 for x in shape]
        point[dimension] = fixed_surface_index
        print( "fsi dimension {} fix_surface_index {} start_plane {}".format(dimension, fixed_surface_index, point))
        other_dimension_iterator(dimension, 0, constraint, halo, shape, point)

    constraint[dimension][0] = halo[dimension]
    constraint[dimension][1] = shape[dimension] - halo[dimension]

def halo_iterator(halo, shape):
    constraint = [[0,size] for size in shape]

    for dimension in range(len(shape)):
        print("constraint {}".format(constraint))
        fixed_surface_iterator(dimension, constraint, halo, shape)


halo_iterator(halo, shape)
