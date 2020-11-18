"""Additional operators, and functions for pruning/simplifying operators."""

# pylint: disable=missing-function-docstring

from collections import OrderedDict

import numpy as np
from nengo.builder.operator import (
    BsrDotInc,
    Copy,
    DotInc,
    ElementwiseInc,
    Operator,
    Reset,
)
from nengo.builder.signal import Signal
from nengo.builder.transforms import ConvInc
from nengo.transforms import SparseMatrix


class MultiDotInc(Operator):
    r"""``y <- gamma + beta * y_in + \sum_i dot(A_i, x_i)``"""

    def __init__(self, Y, Y_in, beta, gamma, tag=None):
        super().__init__(tag=tag)

        assert Y.ndim == 1
        self.Y = Y
        self.Y_in = Y_in
        if Y.shape != Y_in.shape:
            raise TypeError()

        if isinstance(beta, Signal):
            self._float_beta = None
            self._signal_beta = beta
            if beta.shape != Y.shape:
                raise NotImplementedError("", (beta.shape, Y.shape))
        elif hasattr(beta, "value"):
            self._float_beta = float(beta.value)
            self._signal_beta = None
        else:
            self._float_beta = float(beta)
            self._signal_beta = None

        self.gamma = float(gamma)
        self.As = []
        self.Xs = []
        self._incs_Y = (
            self._signal_beta is None and self._float_beta == 1 and self.Y_in is self.Y
        )

    @classmethod
    def convert_to(cls, op):
        if isinstance(op, BsrDotInc):
            raise NotImplementedError(
                "Optimized BsrDotInc operations not yet supported by NengoOCL"
            )
        elif type(op) == DotInc:
            rval = cls(op.Y, op.Y, beta=1, gamma=0, tag=op.tag)
            rval.add_AX(op.A, op.X)
        else:
            return op

        assert set(op.reads).issuperset(rval.reads), (rval.reads, op.reads)
        assert rval.incs == op.incs
        assert rval.sets == op.sets
        assert all(s.size for s in rval.all_signals), op
        assert set(rval.updates) == set(op.updates), (rval.updates, op.updates)
        return rval

    def add_AX(self, A, X):
        assert X.ndim == 1
        self.As.append(A)
        self.Xs.append(X)

    @property
    def reads(self):
        rval = self.As + self.Xs
        if not self._incs_Y:
            if self._signal_beta is None:
                if self._float_beta != 0 and self.Y_in is not self.Y:
                    rval += [self.Y_in]
            else:
                rval += [self._signal_beta]
                if self.Y_in is not self.Y:
                    rval += [self.Y_in]
        return rval

    @property
    def incs(self):
        return [self.Y] if self._incs_Y else []

    @property
    def sets(self):
        return [] if self._incs_Y else [self.Y]

    @property
    def updates(self):
        return []

    def __str__(self):
        beta = self._float_beta if self._signal_beta is None else self._signal_beta
        dots = ["dot(%s, %s)" % (A, X) for A, X in zip(self.As, self.Xs)]
        return (
            "<MultiDotInc(tag=%s, Y=%s, Y_in=%s, beta=%s, gamma=%s, "
            "dots=[%s]) at 0x%x>"
            % (self.tag, self.Y, self.Y_in, beta, self.gamma, ", ".join(dots), id(self))
        )

    def __repr__(self):
        return self.__str__()

    @classmethod
    def compress(cls, operators):
        sets = OrderedDict()
        incs = OrderedDict()
        rval = []
        for op in operators:
            if isinstance(op, cls):
                if op.sets:
                    assert len(op.sets) == 1 and len(op.incs) == 0
                    sets.setdefault(op.sets[0], []).append(op)
                else:
                    assert len(op.incs) == 1 and len(op.sets) == 0
                    incs.setdefault(op.incs[0], []).append(op)
            else:
                rval.append(op)

        # combine incs into sets if on same view
        for view, set_ops in sets.items():
            (set_op,) = set_ops
            inc_ops = incs.get(view, [])
            for inc_op in inc_ops[:]:
                set_op.As.extend(inc_op.As)
                set_op.Xs.extend(inc_op.Xs)
                inc_ops.remove(inc_op)
            rval.append(set_op)

        # combine remaining incs if on same view
        for view, inc_ops in incs.items():
            if len(inc_ops) > 0:
                inc_op0 = inc_ops[0]
                for inc_op in inc_ops[1:]:
                    inc_op0.As.extend(inc_op.As)
                    inc_op0.Xs.extend(inc_op.Xs)
                rval.append(inc_op0)

        return rval

    @staticmethod
    def _as2d(view):
        if view.ndim == 0:
            return view.reshape(1, 1)
        elif view.ndim == 1:
            return view.reshape(view.shape[0], 1)
        elif view.ndim == 2:
            return view
        else:
            raise ValueError("No support for tensors with %d dimensions" % view.ndim)

    def get_views(self):
        Y_view = self._as2d(self.Y)
        Y_in_view = self._as2d(self.Y_in)
        beta_view = (
            self._as2d(self._signal_beta) if self._signal_beta is not None else None
        )

        A_views = []
        X_views = []
        for A, X in zip(self.As, self.Xs):
            X_view = self._as2d(X)
            if A.ndim == 1 and X.ndim == 1:
                A_view = A.reshape((1, A.shape[0]))  # vector dot
            else:
                A_view = self._as2d(A)

            if A_view.shape == (1, 1):
                # -- scalar AX_views can be done as reverse multiplication
                A_view, X_view = X_view, A_view
            elif not (
                X_view.shape[0] == A_view.shape[1]
                and X_view.shape[1] == Y_view.shape[1]
                and A_view.shape[0] == Y_view.shape[0]
            ):
                raise ValueError(
                    "shape mismach (A: %s, X: %s, Y: %s)"
                    % (A.shape, X.shape, self.Y.shape)
                )

            A_views.append(A_view)
            X_views.append(X_view)

        return A_views, X_views, Y_view, Y_in_view, beta_view

    def make_step(self, signals, dt, rng):
        raise NotImplementedError("MultiDotInc is only supported in OCL")


def signal_io_dicts(operators):
    """
    Organizes operators into dictionaries according to the signals they
    set/inc/read/update.

    Copied from Nengo DL. See there for full documentation
    """

    # note: we manually initialize the arrays because we want there to be
    # an entry for all the signal bases, but get an error if we try to
    # access any non-base signals
    sets = {s.base: [] for op in operators for s in op.all_signals}
    incs = {s.base: [] for op in operators for s in op.all_signals}
    reads = {s.base: [] for op in operators for s in op.all_signals}
    updates = {s.base: [] for op in operators for s in op.all_signals}

    for op in operators:
        for s in op.sets:
            sets[s.base].append(op)
        for s in op.incs:
            incs[s.base].append(op)
        for s in op.reads:
            reads[s.base].append(op)
        for s in op.updates:
            updates[s.base].append(op)

    return sets, incs, reads, updates


def remove_unmodified_resets(operators):
    """
    Remove any Reset operators that are targeting a signal that is
    never modified.

    If a signal is reset, but never inced/updated after that, we can just set
    the default signal value to the reset value and remove the reset. Note:
    this wouldn't normally happen, but it can happen if we removed
    some of the incs (e.g. in `.remove_zero_incs`).

    Parameters
    ----------
    operators : list of `~nengo.builder.Operator`
        Operators in the model

    Returns
    -------
    new_operators : list of `~nengo.builder.Operator`
        Modified list of operators
    """

    _, incs, _, updates = signal_io_dicts(operators)

    new_operators = []
    for op in operators:
        if type(op) == Reset:
            target = op.dst
            if len(incs[target.base]) + len(updates[target.base]) == 0:
                target.initial_value.setflags(write=True)
                target.initial_value[...] = op.value
                target.initial_value.setflags(write=False)
            else:
                new_operators.append(op)
        else:
            new_operators.append(op)

    return new_operators


def remove_zero_incs(operators):
    """
    Remove any operators where we know the input (and therefore output) is
    zero.

    If the input to a DotInc/ElementwiseInc/Copy/ConvInc is zero then we know
    that the output of the op will be zero, so we can just get rid of it.

    Parameters
    ----------
    operators : list of `~nengo.builder.Operator`
        Operators in the model

    Returns
    -------
    new_operators : list of `~nengo.builder.Operator`
        Modified list of operators

    Notes
    -----
    Copied from ``nengo_dl/graph_optimizer.py``.
    """

    def all_zero(sig):
        data = sig.initial_value
        if sig.sparse:
            if not isinstance(data, SparseMatrix):
                data = data.tocoo()
            data = data.data

        return np.all(data == 0)

    sets, incs, _, updates = signal_io_dicts(operators)

    new_operators = []
    for op in operators:
        if isinstance(op, (DotInc, ElementwiseInc, Copy, ConvInc)):
            for src in op.reads:
                # check if the input is the output of a Node (in which case the
                # value might change, so we should never get rid of this op).
                # checking the name of the signal seems a bit fragile, but I
                # can't think of a better solution
                if src.name.startswith("<Node"):
                    continue

                # find any ops that modify src
                pred = sets[src.base] + incs[src.base] + updates[src.base]

                # the input (and therefore output) will be zero if the only
                # input is a Reset(0) op, or the only input is a constant
                # signal (not set/inc/updated) that is all zero
                zero_input = (
                    len(pred) == 1
                    and type(pred[0]) == Reset
                    and np.all(pred[0].value == 0)
                ) or (len(pred) == 0 and all_zero(src))
                if zero_input:
                    if len(op.sets) > 0:
                        new_operators.append(Reset(op.sets[0]))
                    break
            else:
                new_operators.append(op)
        else:
            new_operators.append(op)

    return new_operators


def simplify_operators(operators):
    """Apply simplifications to a list of operators, returning a simplified list.

    Applies all the simplifications in ``nengo_ocl.operators.simplifications``.
    """

    for simplification in simplifications:
        operators = simplification(operators)

    return operators


# the simplifications that will be applied. Modify this global variable to change
# what simplifications are applied.
simplifications = [
    remove_unmodified_resets,
    remove_zero_incs,
]
