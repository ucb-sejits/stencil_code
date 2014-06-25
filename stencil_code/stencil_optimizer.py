from ctree.visitors import NodeTransformer, NodeVisitor
from ctree.c.nodes import *
from ctypes import c_int
from copy import deepcopy


def unroll(tree, for_node, factor):
    Unroller(factor, for_node).visit(tree)


class Unroller(NodeTransformer):
    def __init__(self, factor, for_node):
        self.factor = factor
        self.for_node = for_node

    def visit_For(self, for_node):
        if for_node is not self.for_node:
            map(self.visit, for_node.body)
            return for_node
        factor = self.factor
        # Determine the leftover iterations after unrolling
        # TODO: Requires that parent pointers have been fixed
        initial = for_node.init.right.value
        end = for_node.test.right.value
        # TODO: Lt vs LtE problems
        leftover_begin = int((end - initial + 1) / factor) * factor + initial - 1

        new_end = leftover_begin
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
        # TODO: Handling LT vs LTE cases?
        for_node.test = Lt(for_node.init.left.name, new_end)
        for_node.incr = new_incr
        for_node.body = new_body

        if leftover_begin < end:
            return [for_node, leftover_For]
        return for_node


def block_loops(inner, unblocked, block_factor):
    #factors = [self.block_factor for x in self.output_grid_shape]
    #factors[len(self.output_grid_shape)-1] = 1


    # use the helper class below to do the actual blocking.
    blocked = StencilCacheBlocker().block(unblocked, block_factor)

    # need to update inner to point to the innermost in the new blocked version
    inner = FindInnerMostLoop().find(blocked)

    assert (inner != None)
    return [inner, blocked]


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

class StencilCacheBlocker(object):
    """
    Class that takes a tree of perfectly-nested For loops (as in a stencil) and performs standard cache blocking
    on them.  Usage: StencilCacheBlocker().block(tree, factors) where factors is a tuple, one for each loop nest
    in the original tree.
    """
    class StripMineLoopByIndex(NodeTransformer):
        """Helper class that strip mines a loop of a particular index in the nest."""
        def __init__(self, index, factor):
            self.current_idx = -1
            self.target_idx = index
            self.factor = factor
            super(StencilCacheBlocker.StripMineLoopByIndex, self).__init__()

        def visit_For(self, node):
            self.current_idx += 1

            if self.current_idx == self.target_idx:
                return LoopBlocker().loop_block(node, self.factor)
            else:
                return For(node.init,
                           node.test,
                           node.incr,
                           list(map(self.visit, node.body)))


    def block(self, tree, factors):
        """Main method in StencilCacheBlocker.  Used to block the loops in the tree."""
        # first we apply strip mining to the loops given in factors
        for x in range(len(factors)):

            # we may want to not block a particular loop, e.g. when doing Rivera/Tseng blocking
            if factors[x] > 1:
                tree = StencilCacheBlocker.StripMineLoopByIndex(x*2, factors[x]).visit(tree)

        # now we move all the outer strip-mined loops to be outermost
        for x in range(1,len(factors)):
            if factors[x] > 1:
                tree = self.bubble(tree, 2*x, x)

        return tree

    def bubble(self, tree, index, new_index):
        """
        Helper function to 'bubble up' a loop at index to be at new_index (new_index < index)
        while preserving the ordering of the loops between index and new_index.
        """
        for x in xrange(index-new_index):
            tree = LoopSwitcher().switch(tree, index-x-1, index-x)
        return tree

class LoopBlocker(object):
    def loop_block(self, node, block_size):
        outer_incr_name = SymbolRef(node.init.left.name + node.init.left.name)

        new_inner_test = deepcopy(node.test)
        new_inner_test.right = FunctionCall("min", [
                Add(outer_incr_name, Constant(block_size - 1)),
                node.test.right
            ])
        new_inner_for = For(
            Assign(node.init.left, SymbolRef(outer_incr_name)),
            new_inner_test,
            PostInc(SymbolRef(node.init.left.name)),
            node.body)

        newtest = deepcopy(node.test)
        newtest.left = SymbolRef(node.init.left.name + node.init.left.name)

        old_incr = 1 if type(node.incr) is UnaryOp else node.incr.arg
        new_outer_for = For(
            Assign(SymbolRef(node.init.left.name + node.init.left.name,
                             c_int()),
                             node.init.right),
            newtest,
            AddAssign(SymbolRef(node.init.left.name + node.init.left.name),
                Mul(Constant(old_incr), SymbolRef(block_size))),
            [new_inner_for])

        return new_outer_for

class LoopSwitcher(NodeTransformer):
    """
    Class that switches two loops.  The user is responsible for making sure the switching
    is valid (i.e. that the code can still compile/run).  Given two integers i,j this
    class switches the ith and jth loops encountered.
    """


    def __init__(self):
        self.current_loop = -1
        self.saved_first_loop = None
        self.saved_second_loop = None
        super(LoopSwitcher, self).__init__()

    def switch(self, tree, i, j):
        """Switch the i'th and j'th loops in tree."""
        self.first_target = min(i, j)
        self.second_target = max(i, j)

        self.original_ast = tree

        return self.visit(tree)

    def visit_For(self, node):
        self.current_loop += 1

        if self.current_loop == self.first_target:
            # save the loop
            self.saved_first_loop = node
            new_body = list(map(self.visit, node.body))
            assert self.second_target < self.current_loop + 1, 'Tried to switch loops %d and %d but only %d loops available' % (
                self.first_target, self.second_target, self.current_loop + 1)
            # replace with the second loop (which has now been saved)
            return For(self.saved_second_loop.init,
                       self.saved_second_loop.test,
                       self.saved_second_loop.incr,
                       new_body)

        if self.current_loop == self.second_target:
            # save this
            self.saved_second_loop = node
            # replace this
            return For(self.saved_first_loop.init,
                       self.saved_first_loop.test,
                       self.saved_first_loop.incr,
                       node.body)

        return For(node.init,
                   node.test,
                   node.incr,
                   list(map(self.visit, node.body)))
