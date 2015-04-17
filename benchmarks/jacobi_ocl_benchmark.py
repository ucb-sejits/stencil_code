from __future__ import division
__author__ = 'dorthy'

import numpy
from stencil_code.library.jacobi_stencil import Jacobi
import sys
import re
import os

if __name__ == '__main__':  # pragma no cover
    import argparse

    parser = argparse.ArgumentParser("Run jacobi stencil")
    parser.add_argument('-r', '--rows', action="store", dest="rows", type=int, default=10)
    parser.add_argument('-c', '--cols', action="store", dest="cols", type=int, default=10)
    parser.add_argument('-l', '--log', action="store_true", dest="log")
    parser.add_argument('-b', '--backend', action="store", dest="backend", default="c")
    parser.add_argument('-bh', '--boundary_handling', action="store", dest="boundary_handling", default="clamp")
    parser.add_argument('-pr', '--print-rows', action="store", dest="print_rows", type=int, default=-1)
    parser.add_argument('-pc', '--print-cols', action="store", dest="print_cols", type=int, default=-1)
    parser.add_argument('-i', '--iterations', action="store", type=int, default=-1)
    parser.add_argument('-cr', '--clear-result', action="store_true")

    parse_result = parser.parse_args()

    if parse_result.log:
        import logging
        logging.basicConfig(level=20)

    if parse_result.clear_result:
        open("jacobi_local_size.txt", "w").close()

    rows = parse_result.rows
    cols = parse_result.cols
    iterations = parse_result.iterations
    backend = parse_result.backend
    boundary_handling = parse_result.boundary_handling
    print_rows = parse_result.print_rows if parse_result.print_rows >= 0 else min(10, rows)
    print_cols = parse_result.print_cols if parse_result.print_cols >= 0 else min(10, cols)

    in_img = numpy.ones([rows, cols]).astype(numpy.float32)
    stencil = Jacobi(backend=backend, boundary_handling=boundary_handling)

    out = sys.stdout
    f = open("jacobi_raw_output.txt", "w")
    sys.stdout = f

    for trial in range(iterations):
        out_img = stencil(in_img)

    f.close()
    sys.stdout = out
    f = open("jacobi_raw_output.txt", "r")

    results = list()

    for result in f.read().split("\n"):
        result = result.replace("(", "")
        result = result.replace(")", "")
        result = result.replace("{", "")
        result = result.replace("}", "")
        result = result.replace("\'", "")
        result = result.replace(":", "")
        result = result.replace(",", "")
        record = re.match(r'Tuning run result local_work_size (.*) (.*) time (.*)', result)
        if record:
            results.append((record.group(1), record.group(2), record.group(3)))

    curr_results = sorted(results, key=lambda res: res[2])

    f.close()
    f = open("jacobi_local_size.txt", "a")

    f.write("Input Size: {} {}\n".format(rows, cols))
    f.write("Local Work Size:   Time:            Volume/Surface Area Ratio:\n")

    for res in curr_results:
        dim1, dim2, time = int(res[0]), int(res[1]), float(res[2])
        volume = dim1 * dim2
        sa = dim1 + dim1 + dim2 + dim2
        f.write("{0:<5} {1:<12} {2:<16.10f} {3:<0.10f}\n".format(dim1, dim2, time, volume/sa))

    best = str(stencil.specializer._tuner.get_best_configuration())
    best = best.replace("{'local_work_size': (", ", ").replace(",", "").replace(")}", "")
    f.write("Optimal Local Work Size {}\n\n".format(best))
    f.close()
    os.remove("jacobi_raw_output.txt")
