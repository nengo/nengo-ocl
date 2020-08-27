"""This module contains a parser to turn Python functions into OCL code.

.. todo:: Better testing, i.e., write test cases for all the test functions
          at the bottom of this file.

.. todo:: Get binary_and, or, xor, etc. functions working (priority = low)

   * this will require the ability to specify integer input variables
   * or perhaps just cast inputs to these functions to integers

.. todo:: A danger right now is that there is no check that the user uses all
   passed inputs in their function. For example, if the fn is meant to act on
   three arguments, and the user makes a mistake in their model that passes a
   5-vector to the function, no warning is issued. There's no obvious way to
   deal with this better, though.
"""

try:
    import __builtin__
except ImportError:
    # Renamed in Python 3
    import builtins as __builtin__
import ast
import inspect
import math
from collections import OrderedDict

import numpy as np

from nengo.utils.numpy import is_iterable, is_number


def is_symbolic(x):
    return isinstance(x, Expression) or (
        is_iterable(x) and all(isinstance(xx, Expression) for xx in x)
    )


infix_binary_ops = {
    ast.Add: "+",
    ast.And: "&&",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.Div: "/",
    ast.Eq: "==",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Mod: "%",
    ast.Mult: "*",
    ast.NotEq: "!=",
    ast.Or: "||",
    ast.Sub: "-",
}

prefix_unary_ops = {
    ast.Not: "!",
    ast.UAdd: "",
    ast.USub: "-",
}

# list of functions that we can map directly onto OCL (all unary)
direct_funcs = {
    abs: "fabs",
    math.acos: "acos",
    math.acosh: "acosh",
    math.asin: "asin",
    math.asinh: "asinh",
    math.atan: "atan",
    math.atanh: "atanh",
    math.ceil: "ceil",
    math.cos: "cos",
    math.cosh: "cosh",
    math.exp: "exp",
    math.fabs: "fabs",
    math.floor: "floor",
    math.isinf: "isinf",
    math.isnan: "isnan",
    math.log: "log",
    math.log10: "log10",
    math.log1p: "log1p",
    # math.modf: # TODO: return integer and fractional parts of x
    math.sin: "sin",
    math.sinh: "sinh",
    math.sqrt: "sqrt",
    math.tan: "tan",
    math.tanh: "tanh",
    np.abs: "fabs",
    np.absolute: "fabs",
    np.arccos: "acos",
    np.arccosh: "acosh",
    np.arcsin: "asin",
    np.arcsinh: "asinh",
    np.arctan: "atan",
    np.arctanh: "atanh",
    np.ceil: "ceil",
    np.cos: "cos",
    np.cosh: "cosh",
    np.exp: "exp",
    np.exp2: "exp2",
    np.expm1: "expm1",
    np.fabs: "fabs",
    np.floor: "floor",
    np.isfinite: "isfinite",
    np.isinf: "isinf",
    np.isnan: "isnan",
    np.log: "log",
    np.log10: "log10",
    np.log1p: "log1p",
    np.log2: "log2",
    np.sin: "sin",
    np.sinh: "sinh",
    np.sqrt: "sqrt",
    np.tan: "tan",
    np.tanh: "tanh",
}

try:
    # These are only available in Python 2.7+
    direct_funcs[math.expm1] = "expm1"
    direct_funcs[math.erf] = "erf"
    direct_funcs[math.erfc] = "erfc"
    direct_funcs[math.lgamma] = "lgamma"
    direct_funcs[math.expm1] = "expm1"
except AttributeError:
    pass

# List of functions that are supported, but cannot be directly mapped onto
# a unary OCL function
indirect_funcs = {
    math.atan2: lambda x, y: FuncExp("atan2", x, y),
    # math.copysign: TODO,
    math.degrees: lambda x: BinExp(x, "*", BinExp(NumExp(180), "*", VarExp("M_1_PI"))),
    math.fmod: lambda x, y: FuncExp("fmod", x, y),
    math.hypot: lambda x, y: FuncExp(
        "sqrt", BinExp(BinExp(x, "*", x), "+", BinExp(y, "*", y))
    ),
    math.ldexp: lambda x, y: BinExp(x, "*", FuncExp("pow", NumExp(2), y)),
    math.pow: lambda x, y: FuncExp("pow", x, y),
    math.radians: lambda x: BinExp(x, "*", BinExp(VarExp("M_PI"), "/", NumExp(180))),
    np.add: lambda x, y: BinExp(x, "+", y),
    np.arctan2: math.atan2,
    np.asarray: lambda x: x,
    # np.bitwise_and: lambda x, y: BinExp(x, '&', y),
    # np.bitwise_not: lambda x: UnaryExp('~', x),
    # np.bitwise_or: lambda x, y: BinExp(x, '|', y),
    # np.bitwise_xor: lambda x, y: BinExp(x, '^', y),
    # np.copysign: ,
    np.deg2rad: math.radians,
    np.degrees: math.degrees,
    np.divide: lambda x, y: BinExp(x, "/", y),
    np.equal: lambda x, y: BinExp(x, "==", y),
    np.floor_divide: lambda x, y: FuncExp("floor", BinExp(x, "/", y)),
    np.fmax: lambda x, y: FuncExp("fmax", x, y),
    np.fmin: lambda x, y: FuncExp("fmin", x, y),
    np.fmod: math.fmod,
    np.greater: lambda x, y: BinExp(x, ">", y),
    np.greater_equal: lambda x, y: BinExp(x, ">=", y),
    np.hypot: math.hypot,
    # np.invert: lambda x: UnaryExp('~', x),
    np.ldexp: math.ldexp,
    # np.left_shift: lambda x, y: BinExp(x, '<<', y),
    np.less: lambda x, y: BinExp(x, "<", y),
    np.less_equal: lambda x, y: BinExp(x, "<=", y),
    np.logaddexp: lambda x, y: FuncExp(
        "log", BinExp(FuncExp("exp", x), "+", FuncExp("exp", y))
    ),
    np.logaddexp2: lambda x, y: FuncExp(
        "log2", BinExp(FuncExp("exp2", x), "+", FuncExp("exp2", y))
    ),
    # np.logical_and: lambda x, y: BinExp(x, '&&', y),
    # np.logical_not: lambda x: UnaryExp('!', x),
    # np.logical_or: lambda x, y: BinExp(x, '||', y),
    # np.logical_xor: lambda x, y: BinExp(x, '^^', y),
    np.maximum: np.fmax,
    np.minimum: np.fmin,
    np.mod: math.fmod,
    np.multiply: lambda x, y: BinExp(x, "*", y),
    np.negative: lambda x: UnaryExp("-", x),
    # np.nextafter: # TODO,
    np.power: math.pow,
    # np.prod: # TODO: multiplies array els along axis,
    # np.product: np.prod,
    np.rad2deg: math.degrees,
    np.radians: math.radians,
    np.reciprocal: lambda x: BinExp(NumExp(1.0), "/", x),
    np.remainder: lambda x, y: BinExp(
        x, "-", BinExp(FuncExp("floor", BinExp(x, "/", y)), "*", y)
    ),
    np.sign: lambda x: IfExp(
        BinExp(x, "<=", NumExp(0.0)),
        IfExp(BinExp(x, "<", NumExp(0.0)), NumExp(-1.0), NumExp(0.0)),
        NumExp(1.0),
    ),
    np.signbit: lambda x: BinExp(x, "<", NumExp(0)),
    np.square: lambda x: BinExp(x, "*", x),
    np.subtract: lambda x, y: BinExp(x, "-", y),
}

try:
    # This is only available in Python 2.7+
    indirect_funcs[math.gamma] = lambda x: FuncExp("exp", FuncExp("lgamma", x))
except AttributeError:
    pass


def _recurse_binexp(op, x):
    return BinExp(x[0], op, _recurse_binexp(op, x[1:])) if len(x) > 1 else x[0]


def _all_func(x):
    return _recurse_binexp("&&", x)


def _any_func(x):
    return _recurse_binexp("||", x)


def _len_func(x):
    return NumExp(len(x))


def _max_func(x):
    return FuncExp("max", x[0], _max_func(x[1:])) if len(x) > 1 else x[0]


def _min_func(x):
    return FuncExp("min", x[0], _min_func(x[1:])) if len(x) > 1 else x[0]


def _prod_func(x):
    return _recurse_binexp("*", x)


def _sum_func(x):
    return _recurse_binexp("+", x)


vector_funcs = {
    all: _all_func,
    any: _any_func,
    len: _len_func,
    max: _max_func,
    min: _min_func,
    sum: _sum_func,
    np.all: _all_func,
    np.any: _any_func,
    np.max: _max_func,
    np.min: _min_func,
    np.mean: lambda x: BinExp(_sum_func(x), "/", _len_func(x)),
    np.prod: _prod_func,
    np.sum: _sum_func,
}

vector_attr = {
    "all": _all_func,
    "any": _any_func,
    "max": _max_func,
    "min": _min_func,
    "mean": lambda x: BinExp(_sum_func(x), "/", _len_func(x)),
    "prod": _prod_func,
    "sum": _sum_func,
}

OUTPUT_NAME = "OUTPUT__"


class Expression(object):
    """Represents a numerical expression"""

    def _init_expr(self, expr):
        """Initialize ast.Expr, for use in 'simplify'"""
        expr.lineno = 1
        expr.col_offset = 0
        return expr

    def simplify(self):
        return self  # by default, don't simplify

    def to_ocl(self, wrap=False):
        raise NotImplementedError()

    def __str__(self):
        return self.to_ocl()


class VarExp(Expression):
    def __init__(self, name):
        self.name = name

    def to_ocl(self, wrap=False):
        return self.name


class NumExp(Expression):
    def __init__(self, value):
        self.value = value

    def to_ocl(self, wrap=False):
        if isinstance(self.value, float):
            # Append an 'f' to floats, o.w. some calls (e.g. pow) ambiguous
            # TODO: can we get around putting the 'f' afterwards?
            return "%sf" % self.value
        elif isinstance(self.value, bool):
            return "1" if self.value else "0"
        else:
            return str(self.value)


class UnaryExp(Expression):
    def __init__(self, op, right):
        assert isinstance(right, Expression)
        self.op, self.right = op, right

    def simplify(self):
        op, right = self.op, self.right
        if isinstance(right, NumExp) and not isinstance(op, str):
            # simplify and return NumExp
            a = self._init_expr(ast.Num(right.value))
            c = self._init_expr(ast.UnaryOp(op, a))
            return NumExp(eval(compile(ast.Expression(c), "<string>", "eval")))
        else:
            return self

    def to_ocl(self, wrap=False):
        if isinstance(self.op, str):
            op = self.op
        else:
            op = prefix_unary_ops.get(type(self.op))
            if op is None:
                raise NotImplementedError(
                    "'%s' operator is not supported" % type(self.op).__name__
                )

        if isinstance(self.right, NumExp) and self.right.value < 0:
            s = "%s(%s)" % (op, self.right.to_ocl())
        else:
            s = "%s%s" % (op, self.right.to_ocl(wrap=True))
        return ("(%s)" % s) if wrap else s


class BinExp(Expression):
    def __init__(self, left, op, right):
        assert isinstance(right, Expression) and isinstance(left, Expression)
        self.left, self.op, self.right = left, op, right

    def simplify(self):
        left, op, right = self.left, self.op, self.right
        if (
            isinstance(left, NumExp)
            and isinstance(right, NumExp)
            and not isinstance(op, str)
        ):
            # simplify and return NumExp
            a = self._init_expr(ast.Num(left.value))
            b = self._init_expr(ast.Num(right.value))
            if isinstance(self.op, ast.cmpop):
                c = self._init_expr(ast.Compare(a, [op], [b]))
            else:
                c = self._init_expr(ast.BinOp(a, op, b))
            return NumExp(eval(compile(ast.Expression(c), "<string>", "eval")))
        else:
            return self

    def to_ocl(self, wrap=False):
        left, right = self.left, self.right

        if isinstance(self.op, str):
            op = self.op
        else:
            opt = type(self.op)
            op = infix_binary_ops.get(opt)
            if op is None and opt is ast.Pow:
                if isinstance(right, NumExp):
                    if isinstance(right.value, int):
                        if right.value == 2:
                            return BinExp(left, "*", left).to_ocl(wrap=wrap)
                        else:
                            return FuncExp("pown", left, right).to_ocl(wrap=wrap)
                    elif right.value > 0:
                        return FuncExp("powr", left, right).to_ocl(wrap=wrap)
                return FuncExp("pow", left, right).to_ocl(wrap=wrap)
            elif op is None:
                raise NotImplementedError(
                    "'%s' operator is not supported" % opt.__name__
                )

        s = "%s %s %s" % (left.to_ocl(wrap=True), op, right.to_ocl(wrap=True))
        return ("(%s)" % s) if wrap else s


class FuncExp(Expression):
    def __init__(self, fn, *args):
        self.fn = fn
        self.args = args

    def simplify(self):
        is_num = lambda x: isinstance(x, NumExp)
        if isinstance(self.fn, str):
            return self  # cannot simplify
        elif all(map(is_num, self.args)):
            # simplify scalar function
            return NumExp(self.fn(*[a.value for a in self.args]))
        elif all(
            is_num(a) or is_iterable(a) and all(map(is_num, a)) for a in self.args
        ):
            # simplify vector function
            return NumExp(
                self.fn(
                    [
                        [aa.value for aa in a] if is_iterable(a) else a.value
                        for a in self.args
                    ]
                )
            )
        else:
            return self  # cannot simplify

    def to_ocl(self, wrap=False):
        if isinstance(self.fn, str):
            args = [arg.to_ocl() for arg in self.args]
            return "%s(%s)" % (self.fn, ", ".join(args))
        else:
            fn, args = self.fn, self.args
            if fn in vector_funcs:
                converter = vector_funcs[fn]
            elif fn in direct_funcs:
                converter = lambda x: FuncExp(direct_funcs[fn], x)
            elif fn in indirect_funcs:
                converter = fn
                while converter in indirect_funcs:
                    converter = indirect_funcs[converter]
            else:
                raise NotImplementedError(
                    "'%s' function is not supported" % (fn.__name__)
                )

            argcount = converter.__code__.co_argcount
            if argcount != len(args):
                raise NotImplementedError(
                    "'%s' function is not supported for %d arguments"
                    % (fn.__name__, len(args))
                )

            exp = converter(*args)
            return exp.to_ocl(wrap=wrap)


class IfExp(Expression):
    def __init__(self, cond, true, false):
        self.cond, self.true, self.false = cond, true, false

    def simplify(self):
        if isinstance(self.cond, NumExp):
            return self.true if self.cond.value else self.false
        else:
            return self

    def to_ocl(self, wrap=False):
        s = "%s ? %s : %s" % (
            self.cond.to_ocl(wrap=True),
            self.true.to_ocl(wrap=True),
            self.false.to_ocl(wrap=True),
        )
        return ("(%s)" % s) if wrap else s


class Function_Finder(ast.NodeVisitor):
    """Finds a FunctionDef or Lambda in an Abstract Syntax Tree"""

    def __init__(self):
        self.fn_node = None

    def generic_visit(self, stmt):
        if isinstance(stmt, ast.Lambda) or isinstance(stmt, ast.FunctionDef):
            if self.fn_node is None:
                self.fn_node = stmt
            else:
                raise NotImplementedError(
                    "The source code associated with the function "
                    "contains more than one function definition"
                )

        super(self.__class__, self).generic_visit(stmt)


class OCL_Translator(ast.NodeVisitor):
    MAX_VECTOR_LENGTH = 25
    builtins = __builtin__.__dict__

    def _check_vector_length(self, length):
        if length > self.MAX_VECTOR_LENGTH:
            raise ValueError(
                "Vectors of length >%s are not supported" % self.MAX_VECTOR_LENGTH
            )

    def __init__(self, source, globals_dict, closure_dict, in_dims=None, out_dim=None):
        self.source = source
        self.globals = globals_dict
        self.closures = closure_dict

        # self.init: key=local variable name, value=initialization statement
        self.init = OrderedDict()
        self.temp_names = OrderedDict()  # for comprehensions

        # parse and make code
        a = ast.parse(source)
        ff = Function_Finder()
        ff.visit(a)
        function_def = ff.fn_node

        try:
            self.arg_names = [arg.id for arg in function_def.args.args]
        except AttributeError:
            self.arg_names = [arg.arg for arg in function_def.args.args]
        if in_dims is None:
            in_dims = [None] * len(self.arg_names)
        self.arg_dims = in_dims
        self.out_dim = out_dim
        assert len(self.arg_names) == len(self.arg_dims)

        if isinstance(function_def, ast.FunctionDef):
            self.function_name = function_def.name
            self.body = self.visit_block(function_def.body)
        elif isinstance(function_def, ast.Lambda):
            if hasattr(function_def, "targets"):
                self.function_name = function_def.targets[0].id
            else:
                self.function_name = "<lambda>"

            # wrap lambda expression to look like a one-line function
            r = ast.Return()
            r.value = function_def.body
            r.lineno = 1
            r.col_offset = 4
            self.body = self.visit_block([r])
        else:
            raise ValueError(
                "Expected function definition or lambda function assignment, "
                "got " + str(type(function_def))
            )

    def _parse_var(self, var):
        if is_number(var):
            return NumExp(var)
        elif isinstance(var, str):
            return '"%s"' % var
        elif isinstance(var, (list, np.ndarray)):
            if isinstance(var, np.ndarray):
                var = var.tolist()
            self._check_vector_length(len(var))
            return [self._parse_var(v) for v in var]
        else:
            return var

    def visit(self, node):
        if node is None:
            return None

        res = ast.NodeVisitor.visit(self, node)
        return res

    def visit_Name(self, expr):
        name = expr.id
        if name in self.temp_names:
            return self.temp_names[name]
        elif name in self.arg_names:
            dim = self.arg_dims[self.arg_names.index(name)]
            assert (
                dim is not None
            ), "Must provide input dimensionality for vectorized arguments"
            self._check_vector_length(dim)
            return [VarExp("%s[%d]" % (name, i)) for i in range(dim)]
        elif name in self.init:
            return VarExp(name)
        elif name in self.closures:
            return self._parse_var(self.closures[name])
        elif name in self.globals:
            return self._parse_var(self.globals[name])
        elif name in self.builtins:
            return self._parse_var(self.builtins[name])
        else:
            raise ValueError("Unrecognized name '%s'" % name)

    def visit_Num(self, expr):
        return self._parse_var(expr.n)

    def visit_Str(self, expr):
        return self._parse_var(expr.s)

    def _int_index(self, index):
        if index is None:
            return None

        assert isinstance(index, NumExp), "Index must be a number"
        assert isinstance(index.value, int), "Index must be an integer"
        return index.value

    def visit_Index(self, expr):
        index = self.visit(expr.value)
        if isinstance(index, slice):
            # happens if index is a name referring to a slice
            return index

        return self._int_index(index)

    def visit_Ellipsis(self, expr):
        raise NotImplementedError("Ellipsis")

    def visit_Slice(self, expr):
        lower = self._int_index(self.visit(expr.lower))
        upper = self._int_index(self.visit(expr.upper))
        step = self._int_index(self.visit(expr.step))
        return slice(lower, upper, step)

    def visit_ExtSlice(self, expr):
        raise NotImplementedError("ExtSlice")

    def visit_Subscript(self, expr):
        assert isinstance(expr.value, ast.Name)
        var = self.visit(expr.value)
        s = self.visit(expr.slice)
        return var[s]

    def _broadcast_args(self, func, args):
        """Apply 'func' element-wise to lists of args"""
        as_list = lambda x: list(x) if is_iterable(x) else [x]
        args = list(map(as_list, args))
        arg_lens = list(map(len, args))
        max_len = max(arg_lens)
        assert all(n in [0, 1, max_len] for n in arg_lens), (
            "Could not broadcast arguments with lengths %s" % arg_lens
        )

        result = [
            func(*[a[i] if len(a) > 1 else a[0] for a in args]) for i in range(max_len)
        ]
        result = [r.simplify() for r in result]
        return result[0] if len(result) == 1 else result

    def _visit_unary_op(self, op, operand):
        return self._broadcast_args(lambda a: UnaryExp(op, a), [self.visit(operand)])

    def _visit_binary_op(self, op, left, right):
        return self._broadcast_args(
            lambda a, b: BinExp(a, op, b), [self.visit(left), self.visit(right)]
        )

    def visit_UnaryOp(self, expr):
        return self._visit_unary_op(expr.op, expr.operand)

    def visit_BinOp(self, expr):
        return self._visit_binary_op(expr.op, expr.left, expr.right)

    def visit_BoolOp(self, expr):
        if len(expr.values) == 1:
            return self._visit_unary_op(expr.op, expr.values[0])
        elif len(expr.values) == 2:
            return self._visit_binary_op(expr.op, *expr.values)
        else:
            raise NotImplementedError("values > 2 not supported")

    def visit_Compare(self, expr):
        assert len(expr.ops) == 1
        assert len(expr.comparators) == 1
        return self._visit_binary_op(expr.ops[0], expr.left, expr.comparators[0])

    def visit_Call(self, expr):
        assert (
            not expr.keywords and getattr(expr, "kwargs", None) is None
        ), "kwargs not implemented"
        handle = self.visit(expr.func)
        assert callable(handle)
        args = [self.visit(arg) for arg in expr.args]
        if not any(is_symbolic(arg) for arg in args):
            return handle(*args)
        elif handle in vector_funcs:
            return vector_funcs[handle](*args)
        else:
            value = self._broadcast_args(lambda *args: FuncExp(handle, *args), args)
            return value

    def visit_Attribute(self, expr):
        value = self.visit(expr.value)
        if not is_symbolic(value):
            return self._parse_var(getattr(value, expr.attr))
        elif isinstance(value, list) and expr.attr in vector_attr:
            return lambda: vector_attr[expr.attr](value)
        else:
            raise ValueError("Cannot get %r attribute on an expression" % expr.attr)

    def visit_List(self, expr):
        return [self.visit(elt) for elt in expr.elts]

    def visit_Expr(self, expr):
        raise NotImplementedError("Expr")

    def visit_GeneratorExp(self, expr):
        raise NotImplementedError("GeneratorExp")

    def visit_ListComp(self, expr):
        # very limited list comprehension
        assert len(expr.generators) == 1, "Multiple generators not implemented"
        gen = expr.generators[0]
        assert gen.iter.func.id in ["range", "xrange"]
        assert len(gen.iter.args) == 1
        n = self.visit(gen.iter.args[0])
        assert isinstance(n, NumExp)
        assert isinstance(gen.target, ast.Name)
        temp_name = gen.target.id
        assert temp_name not in self.temp_names

        self._check_vector_length(n.value)
        result = []
        for v in range(n.value):
            self.temp_names[temp_name] = NumExp(v)
            result.append(self.visit(expr.elt))

        del self.temp_names[temp_name]
        return result

    def visit_Tuple(self, expr):
        raise NotImplementedError("Tuple")

    def visit_IfExp(self, expr):
        cond = self.visit(expr.test)
        true = self.visit(expr.body)
        false = self.visit(expr.orelse)
        return self._broadcast_args(IfExp, [cond, true, false])

    def visit_Print(self, expr):
        assert expr.dest is None, "other dests not implemented"
        if (
            len(expr.values) == 1
            and isinstance(expr.values[0], ast.BinOp)
            and isinstance(expr.values[0].op, ast.Mod)
            and isinstance(expr.values[0].left, ast.Str)
        ):
            # we're using string formatting
            stmt = self.visit(expr.values[0].left)[:-1] + '\\n"'
            if isinstance(expr.values[0].right, ast.Tuple):
                args = [str(self.visit(arg)) for arg in expr.values[0].right.elts]
            else:
                args = [str(self.visit(expr.values[0].right))]
            return ["printf(%s);" % ", ".join([stmt] + args)]
        else:
            stmt = '"' + " ".join(["%s" for arg in expr.values]) + '\\n"'
            args = ", ".join([str(self.visit(arg)) for arg in expr.values])
            return ["printf(%s, %s);" % (stmt, args)]

    def _visit_lhs(self, lhs):
        if isinstance(lhs, ast.Name):
            name = lhs.id
            unassignables = [self.arg_names, self.globals, self.closures]
            if any(name in d for d in unassignables):
                raise ValueError("Can only assign to a local variable")
            else:
                if name not in self.init:
                    # TODO: make new variables of types other than float?
                    self.init[name] = "float %s;" % name  # make a new variable
                return name
        else:
            raise NotImplementedError("Complex LHS")

    def visit_Assign(self, expr):
        assert len(expr.targets) == 1, "Multiple targets not implemented"
        lhs = self._visit_lhs(expr.targets[0])
        rhs = self.visit(expr.value)
        assert isinstance(
            rhs, Expression
        ), "Can only assign math expressions, not '%s'" % type(rhs)
        return ["%s = %s;" % (lhs, rhs.to_ocl())]

    def visit_AugAssign(self, expr):
        lhs = self._visit_lhs(expr.target)
        rhs = self._visit_binary_op(expr.op, expr.target, expr.value)
        assert isinstance(
            rhs, Expression
        ), "Can only assign math expressions, not '%s'" % type(rhs)
        return ["%s = %s;" % (lhs, rhs.to_ocl())]

    def visit_Return(self, expr):
        value = self.visit(expr.value)
        if is_iterable(value):
            self._check_vector_length(len(value))
            if not all(isinstance(v, Expression) for v in value):
                raise ValueError("Can only return a list of mathematical expressions")
            return [
                "%s[%d] = %s;" % (OUTPUT_NAME, i, v.to_ocl())
                for i, v in enumerate(value)
            ] + ["return;"]
        elif isinstance(value, Expression):
            return ["%s[0] = %s;" % (OUTPUT_NAME, value.to_ocl()), "return;"]
        else:
            raise ValueError(
                "Can only return mathematical expressions, " "or lists of expressions"
            )

    def visit_If(self, expr):
        test = self.visit(expr.test)
        assert not isinstance(
            test, list
        ), "Cannot test vector expression for truth or falsity"
        a = ["if (%s) {" % test, self.visit_block(expr.body)]
        orelse = self.visit_block(expr.orelse)
        if len(orelse) == 0:
            return a + ["}"]
        else:
            return a + ["} else {", orelse, "}"]

    def visit_While(self, expr):
        raise NotImplementedError("While")

    def visit_For(self, expr):
        raise NotImplementedError("For")

    def visit_FunctionDef(self, expr):
        raise NotImplementedError("FunctionDef")

    def visit_Lambda(self, expr):
        raise NotImplementedError("Lambda")

    def visit_block(self, exprs):
        block = []
        for expr in exprs:
            block.extend(self.visit(expr))
        return block


def strip_leading_whitespace(source):
    lines = source.splitlines()
    assert len(lines) > 0
    first_line = lines[0]
    n_removed = len(first_line) - len(first_line.lstrip())
    if n_removed > 0:
        return "\n".join(line[n_removed:] for line in lines)
    else:
        return source


class OCL_Function(object):
    def __init__(self, fn, in_dims=None, out_dim=None):
        if in_dims is not None and not is_iterable(in_dims):
            in_dims = [in_dims]

        self.fn = fn
        self.in_dims = in_dims
        self.out_dim = out_dim
        self._translator = None

    @staticmethod
    def _is_lambda(v):
        return isinstance(v, type(lambda: None)) and v.__name__ == "<lambda>"

    def _get_ocl_translator(self):
        if self.fn in direct_funcs or self.fn in indirect_funcs:
            assert (
                self.in_dims is not None
            ), "Must supply input dimensionality for raw function"
            assert len(self.in_dims) == 1, "Raw functions can only have one input"
            function = self.fn

            def wrapper(x):  # need a wrapper to copy variables
                return function(x)

            fn = wrapper
        else:
            fn = self.fn

        source = inspect.getsource(fn)
        source = strip_leading_whitespace(source)

        try:
            globals_dict = fn.func_globals
            closure_dict = (
                dict(
                    zip(
                        fn.__code__.co_freevars,
                        [c.cell_contents for c in fn.func_closure],
                    )
                )
                if fn.func_closure is not None
                else {}
            )
        except AttributeError:
            globals_dict = fn.__globals__
            closure_dict = (
                {
                    var: contents
                    for var, contents in zip(
                        fn.__code__.co_freevars,
                        [c.cell_contents for c in fn.__closure__],
                    )
                }
                if fn.__closure__ is not None
                else {}
            )

        return OCL_Translator(
            source,
            globals_dict,
            closure_dict,
            in_dims=self.in_dims,
            out_dim=self.out_dim,
        )

    @property
    def translator(self):
        if self._translator is None:
            self._translator = self._get_ocl_translator()
        return self._translator

    def _flatten(self, blocks, indent=0):
        lines = []
        for b in blocks:
            if isinstance(b, list):
                lines.extend(self._flatten(b, indent + 4))
            else:
                lines.append("".join([" "] * indent) + b)
        return lines

    @property
    def init(self):
        return "\n".join(self._flatten(self.translator.init.values()))

    @property
    def code(self):
        return "\n".join(self._flatten(self.translator.body))


if __name__ == "__main__":  # noqa: C901

    def ocl_f(*args, **kwargs):
        ocl_fn = OCL_Function(*args, **kwargs)
        print(ocl_fn.init)
        print(ocl_fn.code)
        print("")
        return ocl_fn

    print("*" * 5 + "Raw" + "*" * 50)
    ocl_f(np.sin, in_dims=(1,))

    print("*" * 5 + "Multi sine" + "*" * 50)
    ocl_f(np.sin, in_dims=(3,))

    print("*" * 5 + "List-return" + "*" * 50)

    def func(t):
        return [1, 2, 3]

    ocl_f(func, in_dims=(1,))

    print("*" * 5 + "Multi-arg" + "*" * 50)

    def func(t, x):
        return t + x[:2] + x[2]

    ocl_f(func, in_dims=(1, 3))

    print("*" * 5 + "Simplify" + "*" * 50)

    def func(y):
        return y + np.sin([1, 2, 3])

    ocl_f(func, in_dims=(1,))

    multiplier = 3842.012

    def square(x):
        print("wow: %f, %d, %s" % (x[0], 9, "hello"))

        if 1 + (2 == 2):
            y = 2.0 * x
            z -= 4 + (3 if x > 99 else 2)  # noqa: F821
        elif x == 2:
            y *= 9.12 if 3 > 4 else 0
            z = 4 * (x - 2)
        else:
            y = 9 * x
            z += x ** (-1.1)

        return np.sin(multiplier * (y * z) + np.square(y))

    ocl_f(square, in_dims=1)

    print("*" * 5 + "Vector lambda" + "*" * 50)
    insert = -0.5
    func = lambda x: x + 3 if all(x > 2) else x - 1
    ocl_f(func, in_dims=3)

    if 0:
        print("*" * 5 + "Large input" + "*" * 50)
        insert = -0.5
        func = lambda x: [x[1] * x[1051], x[3] * x[62]]
        ocl_f(func, in_dims=1100)

    print("*" * 5 + "List comprehension" + "*" * 50)
    insert = -0.5
    func = lambda x: [np.maximum(0.1, np.sin(2)) * x[4 - i] for i in range(5)]
    ocl_f(func, in_dims=5)

    print("*" * 5 + "Unary minus" + "*" * 50)
    insert = -0.5

    def function(x):
        return x * -insert

    ocl_f(function, in_dims=1)

    print("*" * 5 + "Subtract" + "*" * 50)

    def function(x):
        return np.subtract(x[1], x[0])

    ocl_f(function, in_dims=2)

    print("*" * 5 + "List" + "*" * 50)

    def function(y):
        z = y[0] * y[1]
        return [y[1], z]

    ocl_f(function, in_dims=2)

    print("*" * 5 + "Array" + "*" * 50)
    value = np.arange(3)

    def function(y):
        return value

    ocl_f(function, in_dims=value.size)

    print("*" * 5 + "AsArray" + "*" * 50)

    def function(y):
        return np.asarray([y[0], y[1], 3])

    ocl_f(function, in_dims=2)

    print("*" * 5 + "IfExp" + "*" * 50)

    def function(y):
        return 5 if y > 3 else 0

    ocl_f(function, in_dims=1)

    print("*" * 5 + "Sign" + "*" * 50)

    def function(y):
        return np.sign(y)

    ocl_f(function, in_dims=1)

    print("*" * 5 + "Radians" + "*" * 50)
    power = 2

    def function(y):
        return np.radians(y ** power)

    ocl_f(function, in_dims=1)

    print("*" * 5 + "Boolop" + "*" * 50)
    power = 3.2

    def function(y):
        if y > 3 and y < 5:
            return y ** power
        else:
            return np.sign(y)

    ocl_f(function, in_dims=1)

    print("*" * 5 + "Nested return" + "*" * 50)
    power = 3.2

    def function(y):
        if y > 3 and y < 5:
            return y ** power

        return np.sign(y)

    ocl_f(function, in_dims=1)

    print("*" * 5 + "Math constants" + "*" * 50)

    def function(y):
        return np.sin(np.pi * y) + np.e

    ocl_f(function, in_dims=1)

    print("*" * 5 + "Vector functions" + "*" * 50)
    ocl_f(lambda x: x[: len(x) / 2], in_dims=4)
    ocl_f(lambda x: np.sum(x), in_dims=3)
    ocl_f(lambda x: np.mean(x), in_dims=3)
    ocl_f(lambda x: x.min(), in_dims=4)
    ocl_f(lambda x: np.sqrt((x ** 2).mean()), in_dims=5)
