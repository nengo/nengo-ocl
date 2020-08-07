from collections import OrderedDict

from nengo.builder.operator import Operator, BsrDotInc, Copy, DotInc, SparseDotInc
from nengo.builder.signal import Signal
from nengo.builder.transforms import ConvInc
from nengo.version import version_info as nengo_version


class MultiDotInc(Operator):
    """``y <- gamma + beta * y_in + \sum_i dot(A_i, x_i)``"""

    def __init__(self, Y, Y_in, beta, gamma, tag=None):
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
        self.tag = tag
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
