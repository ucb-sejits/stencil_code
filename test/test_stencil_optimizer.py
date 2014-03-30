import unittest

from stencil_code.stencil_optimizer import *
from ctree.c.nodes import *
from ctree.transformations import FixUpParentPointers
from ctree.c.types import *


class TestUnroll(unittest.TestCase):
    def _check(self, actual, expected):
        self.assertEqual(str(actual), str(expected))

    def test_simple_unroll(self):
        actual = For(Assign(SymbolRef('x', Int()), Constant(0)),
                     Lt(SymbolRef('x'), Constant(9)),
                     PostInc(SymbolRef('x')),
                     [Add(Constant(1), Constant(2))]
                     )
        expected = For(Assign(SymbolRef('x', Int()), Constant(0)),
                       Lt(SymbolRef('x'), Constant(9)),
                       AddAssign(SymbolRef('x'), 2),
                       [
                           Add(Constant(1), Constant(2)),
                           Add(Constant(1), Constant(2))
                       ])
        unroll(actual, 2)
        self._check(actual, expected)

    def test_leftover_unroll(self):
        actual = For(Assign(SymbolRef('y', Int()), Constant(0)),
                     Lt(SymbolRef('y'), Constant(10)),
                     PostInc(SymbolRef('y')),
                     [
                         For(Assign(SymbolRef('x', Int()), Constant(0)),
                             Lt(SymbolRef('x'), Constant(10)),
                             PostInc(SymbolRef('x')),
                             [Add(Constant(1), Constant(2))]
                             )
                     ])
        expected = For(Assign(SymbolRef('y', Int()), Constant(0)),
                       Lt(SymbolRef('y'), Constant(10)),
                       PostInc(SymbolRef('y')),
                       [
                           For(Assign(SymbolRef('x', Int()), Constant(0)),
                               Lt(SymbolRef('x'), Constant(9)),
                               AddAssign(SymbolRef('x'), 2),
                               [
                                   Add(Constant(1), Constant(2)),
                                   Add(Constant(1), Constant(2))
                               ]),
                           For(Assign(SymbolRef('x', Int()), Constant(9)),
                               Lt(SymbolRef('x'), Constant(10)),
                               PostInc(SymbolRef('x')),
                               [Add(Constant(1), Constant(2))]
                               )
                       ])
        actual = FixUpParentPointers().visit(actual)
        unroll(FindInnerMostLoop().find(actual), 2)
        self._check(actual, expected)
