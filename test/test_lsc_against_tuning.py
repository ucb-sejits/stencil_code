__author__ = 'dorthy'

from stencil_code.backend.local_size_computer import LocalSizeComputer

import opentuner
from opentuner import ConfigurationManipulator
from opentuner import EnumParameter
from opentuner import Result
from opentuner.api import TuningRunManager
from opentuner.measurement.interface import DefaultMeasurementInterface

import sys
import argparse

# class LocalSizeTuner(MeasurementInterface):
#
#     max_work_group_size = 1
#     shape = [1, 1]
#     sizes_to_try = [(1, 1)]
#     final_config = None
#
#     def manipulator(self):
#         """
#         Define the search space
#         """
#         manipulator = ConfigurationManipulator()
#         manipulator.add_parameter(EnumParameter('size', LocalSizeTuner.sizes_to_try))
#         return manipulator
#
#     def run(self, desired_result, input, limit):
#         """
#         Test a given configuration then
#         return performance
#         """
#         cfg = desired_result.configuration.data
#         volume, surface_area = 1, 0
#         for d in cfg['size']:
#             volume *= d
#             surface_area += d
#         if volume > LocalSizeTuner.max_work_group_size:
#             ratio = float("inf")
#         else:
#             ratio = surface_area / float(volume) #want to minimize this ratio
#         return Result(time=ratio) #this is a hack, need to write actual MinimizeRatio Objective
#     def save_final_config(self, configuration):
#         """called at the end of tuning"""
#         LocalSizeTuner.final_config = configuration.data['size']


def test_func(cfg, max_work_group_size):
    local_work_size = cfg['local_work_size']
    volume, surface_area = 1, 0
    for d in local_work_size:
        volume *= d
        surface_area += d
    if volume > max_work_group_size:
        ratio = float("inf")
    else:
        ratio = surface_area / float(volume)
    return ratio


def main():
    parser = argparse.ArgumentParser(parents=opentuner.argparsers())
    parser.add_argument('dimensions', metavar='D', type=int, nargs='+', help='array dimensions')
    args = parser.parse_args()
    manipulator = ConfigurationManipulator()
    shape = tuple(args.dimensions)
    lsc = LocalSizeComputer(shape)
    bulky_results = lsc.compute_local_size_bulky()
    manipulator.add_parameter(EnumParameter('local_work_size', lsc.get_sizes_tried()))
    interface = DefaultMeasurementInterface(args=args,
                                            manipulator=manipulator,
                                            project_name='examples',
                                            program_name='api_test',
                                            program_version='0.1')
    api = TuningRunManager(interface, args)
    for x in xrange(500):
        desired_result = api.get_next_desired_result()
        if desired_result is None:
          # The search space for this example is very small, so sometimes
          # the techniques have trouble finding a config that hasn't already
          # been tested.  Change this to a continue to make it try again.
          break
        cfg = desired_result.configuration.data
        result = Result(time=test_func(cfg, lsc.max_work_group_size))
        api.report_result(desired_result, result)

    best_cfg = api.get_best_configuration()
    api.finish()
    print "best local_work_size found was {}".format(best_cfg['local_work_size'])
    print "lsc compute bulky : {}".format(bulky_results)
    assert best_cfg['local_work_size'] == bulky_results or \
           best_cfg['local_work_size'] == (bulky_results[1], bulky_results[0])

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
