from ctree.visitors import NodeTransformer, NodeVisitor
from ctree.c.nodes import *
from copy import deepcopy


def unroll(for_node, factor):
    # Determine the leftover iterations after unrolling
    initial = for_node.init.right.value
    end = for_node.test.right.value
    leftover_begin = int((end - initial + 1) / factor) * factor + initial

    new_end = leftover_begin - 1
    new_incr = AddAssign(SymbolRef(for_node.incr.arg.name), factor)
    new_body = for_node.body[:]
    for x in range(1, factor):
        new_extension = deepcopy(for_node.body)
        new_extension = map(UnrollReplacer(for_node.init.left.name,
                                           x).visit, new_extension)
        new_body.extend(new_extension)

    leftover_For = For(Assign(for_node.init.left,
                              Constant(leftover_begin)),
                       for_node.test,
                       for_node.incr,
                       for_node.body)
    for_node.test = LtE(for_node.init.left.name, new_end)
    for_node.incr = new_incr
    for_node.body = new_body

    if not leftover_begin >= end:
        for_node.body.append(leftover_For)


class FindInnerMostLoop(NodeVisitor):
    def __init__(self):
        self.inner_most = None

    def find(self, node):
        self.visit(node)
        return self.inner_most

    def visit_For(self, node):
        self.inner_most = node
        map(self.visit, node.body)


class UnrollReplacer(NodeTransformer):
    def __init__(self, loopvar, incr):
        self.loopvar = loopvar
        self.incr = incr
        self.in_new_scope = False
        self.inside_for = False
        super(UnrollReplacer, self).__init__()

    def visit_SymbolRef(self, node):
        if node.name == self.loopvar:
            return Add(node, Constant(self.incr))
        return SymbolRef(node.name)
