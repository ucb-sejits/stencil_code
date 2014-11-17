__author__ = 'chick'

from benchmarks.stencil_benchmarker import StencilBenchmarker, StencilTest
from stencil_code.library.jacobi_stencil import Jacobi


class Test(StencilTest):
    def __init__(self, name, backend):
        super(Test, self).__init__(name)
        self.backend = backend
        self.stencil = None

    def setup(self):
        self.stencil = Jacobi(backend=self.backend)

    def run(self, input_matrix):
        return self.stencil(input_matrix)


benchmarker = StencilBenchmarker(
    [
        Test("python", "python"),
        Test("c", "c"),
    ]
)

benchmarker.run()