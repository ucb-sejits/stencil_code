__author__ = 'chick'

class MockDevice(object):
    def __init__(self, max_work_group_size=512, max_work_item_sizes=None,
                 max_compute_units=40):
        self.max_work_group_size = max_work_group_size
        self.max_work_item_sizes = max_work_item_sizes if max_work_item_sizes is not None else [512, 512, 512]
        self.max_compute_units = max_compute_units

MockCPU = MockDevice(1024, [1024, 1, 1], 8)
MockIrisPro = MockDevice(512, [512, 512, 512], 40)
