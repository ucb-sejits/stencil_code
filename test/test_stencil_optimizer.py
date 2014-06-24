import unittest

from stencil_code.stencil_optimizer import *
from ctree.c.nodes import *
from ctypes import c_int


class TestUnroll(unittest.TestCase):
    def _check(self, actual, expected):
        self.assertEqual(str(actual), str(expected))

    @unittest.skip("Parent pointers no longer supported")
    def test_simple_unroll(self):
        actual = For(Assign(SymbolRef('x', c_int()), Constant(0)),
                     Lt(SymbolRef('x'), Constant(9)),
                     PostInc(SymbolRef('x')),
                     [Add(Constant(1), Constant(2))]
                     )
        expected = For(Assign(SymbolRef('x', c_int()), Constant(0)),
                       Lt(SymbolRef('x'), Constant(9)),
                       AddAssign(SymbolRef('x'), 2),
                       [
                           Add(Constant(1), Constant(2)),
                           Add(Constant(1), Constant(2))
                       ])
        unroll(actual, 2)
        self._check(actual, expected)

    @unittest.skip("Parent pointers no longer supported")
    def test_leftover_unroll(self):
        actual = For(Assign(SymbolRef('y', c_int()), Constant(0)),
                     Lt(SymbolRef('y'), Constant(10)),
                     PostInc(SymbolRef('y')),
                     [
                         For(Assign(SymbolRef('x', c_int()), Constant(0)),
                             Lt(SymbolRef('x'), Constant(10)),
                             PostInc(SymbolRef('x')),
                             [Add(Constant(1), Constant(2))]
                             )
                     ])
        expected = For(Assign(SymbolRef('y', c_int()), Constant(0)),
                       Lt(SymbolRef('y'), Constant(10)),
                       PostInc(SymbolRef('y')),
                       [
                           For(Assign(SymbolRef('x', c_int()), Constant(0)),
                               Lt(SymbolRef('x'), Constant(9)),
                               AddAssign(SymbolRef('x'), 2),
                               [
                                   Add(Constant(1), Constant(2)),
                                   Add(Constant(1), Constant(2))
                               ]),
                           For(Assign(SymbolRef('x', c_int()), Constant(9)),
                               Lt(SymbolRef('x'), Constant(10)),
                               PostInc(SymbolRef('x')),
                               [Add(Constant(1), Constant(2))]
                               )
                       ])
        unroll(FindInnerMostLoop().find(actual), 2)
        self._check(actual, expected)
