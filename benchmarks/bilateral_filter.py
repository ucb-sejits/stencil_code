__author__ = 'chick'

from benchmarks.stencil_benchmarker import StencilBenchmarker, StencilTest, PrimedStencilTest
from stencil_code.library.better_bilateral_filter import BetterBilateralFilter


benchmarker = StencilBenchmarker(
    [
        # StencilTest("python", "python"),
        StencilTest("unprimed-c", stencil=BetterBilateralFilter, backend='c'),
        PrimedStencilTest("primed-c", stencil=BetterBilateralFilter, backend='c'),
        StencilTest("unprimed-ocl", stencil=BetterBilateralFilter, backend='ocl'),
        PrimedStencilTest("primed-ocl", stencil=BetterBilateralFilter, backend='ocl'),
    ]
)

benchmarker.run()