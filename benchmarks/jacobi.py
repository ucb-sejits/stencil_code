__author__ = 'chick'

from benchmarks.stencil_benchmarker import StencilBenchmarker, StencilTest, PrimedStencilTest
from stencil_code.library.jacobi_stencil import Jacobi


benchmarker = StencilBenchmarker(
    [
        # StencilTest("python", "python"),
        StencilTest("unprimed-c", stencil=Jacobi, backend='c'),
        PrimedStencilTest("primed-c", stencil=Jacobi, backend='c'),
        StencilTest("unprimed-ocl", stencil=Jacobi, backend='ocl'),
        PrimedStencilTest("primed-ocl", stencil=Jacobi, backend='ocl'),
    ]
)

benchmarker.run()