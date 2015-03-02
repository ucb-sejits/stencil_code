__author__ = 'dorthy'

from stencil_code.backend.local_size_computer import LocalSizeComputer

import opentuner
from opentuner import ConfigurationManipulator
from opentuner import EnumParameter
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result
import sys

class LocalSizeTuner(MeasurementInterface):

    max_work_group_size = 1
    shape = [1, 1]
    sizes_to_try = [(1, 1)]

    def manipulator(self):
        """
        Define the search space
        """
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(EnumParameter('size', LocalSizeTuner.sizes_to_try))
        return manipulator

    def run(self, desired_result, input, limit):
        """
        Test a given configuration then
        return performance
        """
        cfg = desired_result.configuration.data
        volume, surface_area = 1, 0
        for d in cfg['size']:
            volume *= d
            surface_area += d
        if volume > LocalSizeTuner.max_work_group_size:
            ratio = float("inf")
        else:
            ratio = surface_area / float(volume) #want to minimize this ratio
        return Result(time=ratio) #this is a hack, need to write actual MinimizeRatio Objective

if __name__ == '__main__':
    if sys.argv[1:]:
        shape = [int(arg) for arg in sys.argv[1:]]
    else:
        shape = [64, 64]
    lsc = LocalSizeComputer(shape)
    print "shape: ", shape
    print "max local group sizes: ", lsc.max_local_group_sizes
    print "max work group size: ",lsc.max_work_group_size
    print "compute units: ", lsc.compute_units
    bulky_results = lsc.compute_local_size_bulky()
    print "lsc compute bulky :", bulky_results[0]
    LocalSizeTuner.max_work_group_size = lsc.max_work_group_size
    LocalSizeTuner.shape = shape
    LocalSizeTuner.sizes_to_try = bulky_results[1]

    argparser = opentuner.default_argparser()
    x = LocalSizeTuner.main(argparser.parse_args())
    print x

    print "done"
