__author__ = 'dorthyluu'

from ctree.opentuner.driver import OpenTunerDriver, CtreeMeasurementInterface

import threading
import argparse
import Queue as queue # queue in 3.x

from ctree import CONFIG
from ctree.tune import TuningDriver

import opentuner
from opentuner.measurement import MeasurementInterface
from opentuner.resultsdb.models import Result
from opentuner.tuningrunmain import TuningRunMain
from opentuner.search.manipulator import ConfigurationManipulator
from opentuner.measurement.inputmanager import FixedInputManager
from opentuner.api import TuningRunManager


class StencilOpenTunerDriver(OpenTunerDriver):

    def reset(self, *ot_args, **ot_kwargs):
        """
        Creates communication queues and spawn a thread
        to run the tuning logic.
        """
        # super(OpenTunerDriver, self).__init__()
        self._best_config = None
        interface = CtreeMeasurementInterface(self, *ot_args, **ot_kwargs)
        arg_parser = argparse.ArgumentParser(parents=opentuner.argparsers())
        config_args = CONFIG.get("opentuner", "args").split()
        tuner_args = arg_parser.parse_args(config_args)
        self.manager = TuningRunManager(interface, tuner_args)
        self._converged = False