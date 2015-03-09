__author__ = 'dorthy'

import argparse
import numpy
from ctree.util import Timer

import opentuner
from opentuner import ConfigurationManipulator
from opentuner import EnumParameter
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result
from opentuner.api import TuningRunManager
from opentuner.measurement.interface import DefaultMeasurementInterface
from stencil_code.backend.local_size_computer import LocalSizeComputer
from stencil_code.library.laplacian import LaplacianKernel


def test_func(cfg, max_work_group_size):
    # return cfg['local_work_size'][0]
    local_work_size = cfg['local_work_size']
    volume, surface_area = 1, 0
    for d in local_work_size:
        volume *= d
        surface_area += d
    if volume > max_work_group_size:
        ratio = float("inf")
    else:
        ratio = surface_area / float(volume) #want to minimize this ratio
    return ratio


def main():
    parser = argparse.ArgumentParser(parents=opentuner.argparsers())
    parser.add_argument('dimensions', metavar='D', type=int, nargs='+', help='array dimensions')
    args = parser.parse_args()
    shape = tuple(args.dimensions)

    input_grid = numpy.random.random(shape).astype(numpy.float32) * 1024

    laplacian = LaplacianKernel(backend='ocl')

    manipulator = ConfigurationManipulator()

    lsc = LocalSizeComputer(shape)
    bulky_results = lsc.compute_local_size_bulky()
    sizes_to_try = bulky_results[1]
    manipulator.add_parameter(EnumParameter('local_work_size', sizes_to_try))
    interface = DefaultMeasurementInterface(args=args,
                                            manipulator=manipulator,
                                            project_name='examples',
                                            program_name='api_test',
                                            program_version='0.1')
    api = TuningRunManager(interface, args)
    # for x in xrange(len(sizes_to_try)):
    for x in range(1):
        desired_result = api.get_next_desired_result()
        if desired_result is None:
          # The search space for this example is very small, so sometimes
          # the techniques have trouble finding a config that hasn't already
          # been tested.  Change this to a continue to make it try again.
          break
        cfg = desired_result.configuration.data
        laplacian.forced_local_size = cfg['local_work_size']
        a = laplacian(input_grid)
        with Timer() as s_t:
            for _ in range(5):
                a = laplacian(input_grid)
        print("{} {} Specialized time".format(s_t.interval, laplacian.forced_local_size))
        print("a.shape {}".format(a.shape))
        result = Result(time=s_t.interval)
        api.report_result(desired_result, result)

    best_cfg = api.get_best_configuration()
    api.finish()
    print 'best local_work_size found was', best_cfg['local_work_size']
    print "lsc compute bulky :", bulky_results[0]

if __name__ == '__main__':
    main()



# if __name__ == '__main__':
#     # if sys.argv[1:]:
#     #     shape = [int(arg) for arg in sys.argv[1:]]
#     # else:
#     shape = [64, 128]
#     lsc = LocalSizeComputer(shape)
#     # print "shape: ", shape
#     # print "max local group sizes: ", lsc.max_local_group_sizes
#     # print "max work group size: ",lsc.max_work_group_size
#     # print "compute units: ", lsc.compute_units
#     bulky_results = lsc.compute_local_size_bulky()
#     LocalSizeTuner.max_work_group_size = lsc.max_work_group_size
#     LocalSizeTuner.shape = shape
#     LocalSizeTuner.sizes_to_try = bulky_results[1]
#
#     argparser = opentuner.default_argparser()
#     LocalSizeTuner.main(argparser.parse_args())
#     print "opentuner result ", LocalSizeTuner.final_config
#     print "lsc compute bulky :", bulky_results[0]
#     assert LocalSizeTuner.final_config == bulky_results[0]
