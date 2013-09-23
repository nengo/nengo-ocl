"""
This file holds a parser to turn simple Python functions into OCL code

TODO:
"""

import inspect, ast, _ast, collections
import numpy as np
import math

infix_binary_ops = {
    ast.Add: '+',
    ast.And: '&&',
    ast.BitAnd: '&',
    ast.BitOr: '|',
    ast.BitXor: '^',
    ast.Div: '/',
    ast.Eq: '==',
    ast.Gt: '>',
    ast.GtE: '>=',
    ast.Lt: '<',
    ast.LtE: '<=',
    ast.Mod: '%',
    ast.Mult: '*',
    ast.NotEq: '!=',
    ast.Or: '||',
    ast.Sub: '-',
    }

prefix_unary_ops = {
    ast.Not: '!',
    ast.UAdd: '',
    ast.USub: '-',
    }

### list of functions that we can map directly onto OCL (all unary)
direct_funcs = {
    math.acos: 'acos',
    math.acosh: 'acosh',
    math.asin: 'asin',
    math.asinh: 'asinh',
    math.atan: 'atan',
    math.atanh: 'atanh',
    math.ceil: 'ceil',
    math.cos: 'cos',
    math.cosh: 'cosh',
    math.erf: 'erf',
    math.erfc: 'erfc',
    math.exp: 'exp',
    math.expm1: 'expm1',
    math.fabs: 'fabs',
    math.floor: 'floor',
    math.isinf: 'isinf',
    math.isnan: 'isnan',
    math.lgamma: 'lgamma',
    math.log: 'log',
    math.log10: 'log10',
    math.log1p: 'log1p',
    # math.modf: # TODO: return integer and fractional parts of x
    math.sin: 'sin',
    math.sinh: 'sinh',
    math.sqrt: 'sqrt',
    math.tan: 'tan',
    math.tanh: 'tanh',
    np.abs: 'fabs',
    np.absolute: 'fabs',
    np.arccos: 'acos',
    np.arccosh: 'acosh',
    np.arcsin: 'asin',
    np.arcsinh: 'asinh',
    np.arctan: 'atan',
    np.arctanh: 'atanh',
    np.ceil: 'ceil',
    np.cos: 'cos',
    np.cosh: 'cosh',
    np.exp: 'exp',
    np.exp2: 'exp2',
    np.expm1: 'expm1',
    np.fabs: 'fabs',
    np.floor: 'floor',
    np.isfinite: 'isfinite',
    np.isinf: 'isinf',
    np.isnan: 'isnan',
    np.log: 'log',
    np.log10: 'log10',
    np.log1p: 'log1p',
    np.log2: 'log2',
    np.sin: 'sin',
    np.sinh: 'sinh',
    np.sqrt: 'sqrt',
    np.tan: 'tan',
    np.tanh: 'tanh',
    }

### List of functions that are supported, but cannot be directly mapped onto
### a unary OCL function
indirect_funcs = {
    math.atan2: lambda x, y: FuncExp('atan2', x, y),
    # math.copysign: TODO,
    math.degrees: lambda x: BinExp(
        x, '*', BinExp(NumExp(180), '*', VarExp('M_1_PI'))),
    math.fmod: lambda x, y: FuncExp('fmod', x, y),
    math.gamma: lambda x: FuncExp('exp', FuncExp('lgamma', x)),
    math.hypot: lambda x, y: FuncExp(
        'sqrt', BinExp(BinExp(x, '*', x), '+', BinExp(y, '*', y))),
    math.ldexp: lambda x, y: BinExp(x, '*', FuncExp('pow', NumExp(2), y)),
    math.pow: lambda x, y: FuncExp('pow', x, y),
    math.radians: lambda x: BinExp(
        x, '*', BinExp(VarExp('M_PI'), '/', NumExp(180))),
    np.add: lambda x, y: BinExp(x, '+', y),
    np.arctan2: math.atan2,
    np.asarray: lambda x: x,
    np.bitwise_and: lambda x, y: BinExp(x, '&', y),
    np.bitwise_not: lambda x: UnaryExp('~', x),
    np.bitwise_or: lambda x, y: BinExp(x, '|', y),
    np.bitwise_xor: lambda x, y: BinExp(x, '^', y),
    # np.copysign: ,
    np.deg2rad: math.radians,
    np.degrees: math.degrees,
    np.divide: lambda x, y: BinExp(x, '/', y),
    np.equal: lambda x, y: BinExp(x, '==', y),
    np.floor_divide: lambda x, y: FuncExp('floor', BinExp(x, '/', y)),
    np.fmax: lambda x, y: FuncExp('fmax', x, y),
    np.fmin: lambda x, y: FuncExp('fmin', x, y),
    np.fmod: math.fmod,
    np.greater: lambda x, y: BinExp(x, '>', y),
    np.greater_equal: lambda x, y: BinExp(x, '>=', y),
    np.hypot: math.hypot,
    np.invert: lambda x: UnaryExp('~', x),
    np.ldexp: math.ldexp,
    np.left_shift: lambda x, y: BinExp(x, '<<', y),
    np.less: lambda x, y: BinExp(x, '<', y),
    np.less_equal: lambda x, y: BinExp(x, '<=', y),
    np.logaddexp: lambda x, y: FuncExp(
            'log', BinExp(FuncExp('exp', x), '+', FuncExp('exp', y))),
    np.logaddexp2: lambda x, y: FuncExp(
            'log2', BinExp(FuncExp('exp2', x), '+', FuncExp('exp2', y))),
    np.logical_and: lambda x, y: BinExp(x, '&&', y),
    np.logical_not: lambda x: UnaryExp('!', x),
    np.logical_or: lambda x, y: BinExp(x, '||', y),
    np.logical_xor: lambda x, y: BinExp(x, '^^', y),
    np.maximum: np.fmax,
    np.minimum: np.fmin,
    np.mod: math.fmod,
    np.multiply: lambda x, y: BinExp(x, '*', y),
    np.negative: lambda x: UnaryExp('-', x),
    # np.nextafter: # TODO,
    np.power: math.pow,
    # np.prod: # TODO: multiplies array els along axis,
    # np.product: np.prod,
    np.rad2deg: math.degrees,
    np.radians: math.radians,
    np.reciprocal: lambda x: BinExp(NumExp(1.), '/', x),
    np.remainder: math.fmod,
    np.sign: lambda x: IfExp(
        BinExp(x, '<=', NumExp(0)),
        IfExp(BinExp(x, '<', NumExp(0)), NumExp(-1), NumExp(0)), NumExp(1)),
    np.signbit: lambda x: BinExp(x, '<', NumExp(0)),
    np.square: lambda x: BinExp(x, '*', x),
    np.subtract: lambda x, y: BinExp(x, '-', y),
    }

INPUT_NAME = "__INPUT__"
OUTPUT_NAME = "__OUTPUT__"

class Expression(object):
    """Represents a numerical expression"""
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
            ### Append an 'f' to floats, o.w. some calls (e.g. pow) ambiguous
            ### TODO: can we get around putting the 'f' afterwards?
            return "%sf" % self.value
        else:
            return str(self.value)

class UnaryExp(Expression):
    def __init__(self, op, right):
        self.op, self.right = op, right

    def to_ocl(self, wrap=False):
        if isinstance(self.right, NumExp) and self.right.value < 0:
            s = "%s(%s)" % (self.op, self.right.to_ocl())
        else:
            s = "%s%s" % (self.op, self.right.to_ocl(wrap=True))
        return ("(%s)" % s) if wrap else s

class BinExp(Expression):
    def __init__(self, left, op, right):
        self.left, self.op, self.right = left, op, right

    def to_ocl(self, wrap=False):
        s = "%s %s %s" % (
            self.left.to_ocl(wrap=True), self.op, self.right.to_ocl(wrap=True))
        return ("(%s)" % s) if wrap else s

class FuncExp(Expression):
    def __init__(self, fn, *args):
        self.fn, self.args = fn, args

    def to_ocl(self, wrap=False):
        args = [arg.to_ocl() for arg in self.args]
        return "%s(%s)" % (self.fn, ', '.join(args))

class IfExp(Expression):
    def __init__(self, cond, true, false):
        self.cond, self.true, self.false = cond, true, false

    def to_ocl(self, wrap=False):
        s = "%s ? %s : %s" % (
            self.cond.to_ocl(wrap=True), self.true.to_ocl(wrap=True),
            self.false.to_ocl(wrap=True))
        return ("(%s)" % s) if wrap else s


class Function_Finder(ast.NodeVisitor):
    # Finds a FunctionDef or Lambda in an Abstract Syntax Tree

    def __init__(self):
        self.fn_node = None

    def generic_visit(self, stmt):
        if isinstance(stmt, _ast.Lambda) or isinstance(stmt, _ast.FunctionDef):
            if self.fn_node is None:
                self.fn_node = stmt
            else:
                raise NotImplementedError("The source code associated with the function contains more than one function definition")

        super(self.__class__, self).generic_visit(stmt)


class OCL_Translator(ast.NodeVisitor):
    def __init__(self, source, globals_dict, closure_dict, filename=None):
        self.source = source
        self.globals = globals_dict
        self.closures = closure_dict
        self.filename = filename

        ### parse and make code
        a = ast.parse(source)
        ff = Function_Finder()
        ff.visit(a);
        function_def = ff.fn_node

        if isinstance(function_def, _ast.FunctionDef):
            self.function_name = function_def.name
            self.arg_names = [arg.id for arg in function_def.args.args]
            self.body = self.visit_block(function_def.body)
        elif isinstance(function_def, _ast.Lambda):
            if hasattr(function_def, 'targets'):
                self.function_name = function_def.targets[0].id
            else:
                self.function_name = "<lambda>"

            self.arg_names = [arg.id for arg in function_def.args.args]
            r = _ast.Return() #wrap lambda expression to look like a one-line function
            r.value = function_def.body
            r.lineno = 1
            r.col_offset = 4
            self.body = self.visit_block([r])
        else:
            raise RuntimeError("Expected function definition or lambda function assignment, got " + str(type(function_def)))

        self.filename = filename
        self.init = collections.OrderedDict()

    def _parse_var(self, var):
        if isinstance(var, (float, int)):
            return NumExp(var)
        elif isinstance(var, str):
            return '"%s"' % var
        elif isinstance(var, (list, np.ndarray)):
            if isinstance(var, np.ndarray):
                var = var.tolist()
            return [self._parse_var(v) for v in var]
        else:
            raise NotImplementedError(
                "Python objects of type %s are not supported" %
                var.__class__.__name__)

    def visit(self, node):
        # print "visiting " + node.__class__.__name__
        res = ast.NodeVisitor.visit(self, node)
        return res

    def visit_Name(self, expr):
        name = expr.id
        if name in self.arg_names:
            return VarExp('%s[0]' % name)
        elif name in self.closures:
            return self._parse_var(self.closures[name])
        elif name in self.globals:
            return self._parse_var(self.globals[name])
        elif name in self.init:
            return VarExp(name)
        else:
            raise ValueError("Unrecognized name '%s'" % name)

    def visit_Num(self, expr):
        return self._parse_var(expr.n)

    def visit_Str(self, expr):
        return self._parse_var(expr.s)

    def visit_Index(self, expr):
        value = self.visit(expr.value)
        assert (isinstance(value, NumExp) and isinstance(value.value, int)), (
            "Only integer indices allowed")
        return value.to_ocl()

    def visit_Ellipsis(self, expr):
        raise NotImplementedError("Ellipsis")

    def visit_Slice(self, expr):
        raise NotImplementedError("Slice")

    def visit_ExtSlice(self, expr):
        raise NotImplementedError("ExtSlice")

    def _visit_unary_op(self, op, operand):
        opt = type(op)
        value = self.visit(operand)
        if opt in prefix_unary_ops:
            return UnaryExp(prefix_unary_ops[opt], value)
        else:
            raise NotImplementedError(
                "'%s' operator is not supported" % opt.__name__)

    def _visit_binary_op(self, op, left, right):
        s_left = self.visit(left)
        s_right = self.visit(right)

        opt = type(op)
        if opt in infix_binary_ops:
            return BinExp(s_left, infix_binary_ops[opt], s_right)
        elif opt is ast.Pow:
            if isinstance(s_right, NumExp):
                if isinstance(s_right.value, int):
                    if s_right.value == 2:
                        return BinExp(s_left, '*', s_left)
                    else:
                        return FuncExp("pown", s_left, s_right)
                elif s_right.value > 0:
                    return FuncExp("powr", s_left, s_right)
            return FuncExp("pow", s_left, s_right)
        else:
            raise NotImplementedError(
                "'%s' operator is not supported" % opt.__name__)

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
        return self._visit_binary_op(
            expr.ops[0], expr.left, expr.comparators[0])

    def visit_Subscript(self, expr):
        assert isinstance(expr.value, ast.Name)
        index = self.visit(expr.slice)
        return VarExp("%s[%s]" % (expr.value.id, index))

    def _get_handle(self, expr):
        """Used to get handle on attribute or function"""
        if isinstance(expr, ast.Name):
            return (self.closures[expr.id] if expr.id in self.closures
                    else self.globals[expr.id])
        else:
            return getattr(self._get_handle(expr.value), expr.attr)

    def visit_Call(self, expr):
        assert expr.kwargs is None, "kwargs not implemented"
        handle = self._get_handle(expr.func)
        args = [self.visit(arg) for arg in expr.args]

        if handle in direct_funcs and len(args) == 1:
            return FuncExp(direct_funcs[handle], args[0])
        elif handle in indirect_funcs:
            indirect = handle
            while indirect in indirect_funcs:
                indirect = indirect_funcs[indirect]
            if indirect.func_code.co_argcount == len(args):
                return indirect(*args)

        raise NotImplementedError(
            "'%s' function is not supported for %d arguments"
            % (handle.__name__, len(args)))

    def visit_Attribute(self, expr):
        handle = self._get_handle(expr)
        return self._parse_var(handle)

    def visit_List(self, expr):
        return [self.visit(elt) for elt in expr.elts]

    def visit_Expr(self, expr):
        raise NotImplementedError("Expr")

    def visit_GeneratorExp(self, expr):
        raise NotImplementedError("GeneratorExp")

    def visit_ListComp(self, expr):
        raise NotImplementedError("ListComp")

    def visit_Tuple(self, expr):
        raise NotImplementedError("Tuple")

    def visit_IfExp(self, expr):
        cond = self.visit(expr.test)
        true = self.visit(expr.body)
        false = self.visit(expr.orelse)
        return IfExp(cond, true, false)

    def visit_Print(self, expr):
        assert expr.dest is None, "other dests not implemented"
        if (len(expr.values) == 1
            and isinstance(expr.values[0], ast.BinOp)
            and isinstance(expr.values[0].op, ast.Mod)
            and isinstance(expr.values[0].left, ast.Str)):
            # we're using string formatting
            stmt = self.visit(expr.values[0].left)[:-1] + '\\n"'
            if isinstance(expr.values[0].right, ast.Tuple):
                args = [str(self.visit(arg)) for arg in expr.values[0].right.elts]
            else:
                args = [str(self.visit(expr.values[0].right))]
            return ["printf(%s);" % ', '.join([stmt] + args)]
        else:
            stmt = '"' + ' '.join(['%s' for arg in expr.values]) + '\\n"'
            args = ', '.join([str(self.visit(arg)) for arg in expr.values])
            return ["printf(%s, %s);" % (stmt, args)]

    def visit_lhs(self, lhs):
        if isinstance(lhs, ast.Name):
            name = lhs.id
            if name in self.arg_names or name in self.globals:
                raise ValueError("Cannot assign to arg or global")
            else:
                if name not in self.init:
                    # TODO: make new variables of types other than float?
                    self.init[name] = "float %s;" % name  # make a new variable
                return name
        else:
            raise NotImplementedError("Complex LHS")

    def visit_Assign(self, expr):
        assert len(expr.targets) == 1, "Multiple targets not implemented"
        lhs = self.visit_lhs(expr.targets[0])
        rhs = self.visit(expr.value)
        assert isinstance(rhs, Expression), "Can only assign math expressions"
        return ["%s = %s;" % (lhs, rhs.to_ocl())]

    def visit_AugAssign(self, expr):
        new_value = self.visit_lhs(expr.target)
        target = self._visit_binary_op(expr.op, expr.target, expr.value)
        return ["%s = %s;" % (new_value, target)]

    def visit_Return(self, expr):
        value = self.visit(expr.value)
        if isinstance(value, list):
            for v in value:
                if not isinstance(v, Expression):
                    raise ValueError(
                        "Can only return list of mathematical expressions")
            return ["%s[%d] = %s;" % (OUTPUT_NAME, i, v.to_ocl())
                    for i, v in enumerate(value)] + ["return;"]
        elif isinstance(value, Expression):
            return ["%s[0] = %s;" % (OUTPUT_NAME, value.to_ocl()), "return;"]
        else:
            raise ValueError("Can only return mathematical expressions, "
                             "or lists of expressions")

    def visit_If(self, expr):
        a = ["if (%s) {" % self.visit(expr.test), self.visit_block(expr.body)]
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
        return '\n'.join(line[n_removed:] for line in lines)
    else:
        return source

class OCL_Function(object):
    def __init__(self, fn):
        self.fn = fn
        # self.__name__ = self.fn.__name__
        self._translator = None

    @staticmethod
    def _is_lambda(v):
        return isinstance(v, type(lambda: None)) and v.__name__ == '<lambda>'

    def get_ocl_translator(self):
        # if self._is_lambda(self.fn):
        #     raise NotImplementedError("No lambda functions")
        # elif self.fn in direct_funcs or self.fn in indirect_funcs:

        if self.fn in function_map:
            function = self.fn
            def dummy(x):
                return function(x)
            fn = dummy
        else:
            fn = self.fn

        source = inspect.getsource(fn)
        source = strip_leading_whitespace(source)
        # filename = inspect.getsourcefile(fn)

        globals_dict = fn.func_globals
        closure_dict = (
            dict(zip(fn.func_code.co_freevars,
                     [c.cell_contents for c in fn.func_closure]))
            if fn.func_closure is not None else {})

        return OCL_Translator(source, globals_dict, closure_dict)

        # try:
        #     return OCL_Translator(source, globals_dict, free_vars,
        #                           closure_cells, filename=filename)
        # # except AssertionError as e:
        # #     print "Could not translate to OCL: %s" % e.strerror
        # # except NotImplementedError as e:
        # except Exception as e:
        #     print "Could not translate to OCL: %s" % e.message
        # return None

    @property
    def translator(self):
        if self._translator is None:
            self._translator = self.get_ocl_translator()
        return self._translator

    # @property
    # def can_translate(self):
    #     return self.translator is not None

    def _flatten(self, blocks, indent=0):
        lines = []
        for b in blocks:
            if isinstance(b, list):
                lines.extend(self._flatten(b, indent+4))
            else:
                lines.append("".join([" "]*indent) + b)
        return lines

    @property
    def init(self):
        return '\n'.join(self._flatten(self.translator.init.values()))

    @property
    def code(self):
        return '\n'.join(self._flatten(self.translator.body))


if __name__ == '__main__':

    multiplier = 3842.012
    @OCL_Function
    def square(x):
        print "wow: %f, %d, %s" % (0.3, 9, "hello")
        if x < 0.5 - 0.1:
            y = 2. * x
            z -= 4 + (3 if x > 99 else 2)
        elif x == 2:
            y *= 9.12
            z = 4*(x - 2)
        else:
            y = 9*x
            z += x**(-1.1)

        return np.sin(multiplier * (y * z) + np.square(y))

    # print square(4)
    print square.init
    print square.code

    print '*' * 5 + 'Unary minus' + '*' * 50
    insert = -0.5
    def function(x):
        return x * -insert

    ocl_fn = OCL_Function(function)
    print ocl_fn.init
    print ocl_fn.code

    print '*' * 5 + 'Subtract' + '*' * 50
    def function(x):
        return np.subtract(x[1], x[0])

    ocl_fn = OCL_Function(function)
    print ocl_fn.init
    print ocl_fn.code

    print '*' * 5 + 'List' + '*' * 50
    def function(y):
        z = y[0] * y[1]
        return [y[1], z]

    ocl_fn = OCL_Function(function)
    print ocl_fn.init
    print ocl_fn.code

    print '*' * 5 + 'Array' + '*' * 50
    value = np.arange(3)
    def function(y):
        return value

    ocl_fn = OCL_Function(function)
    print ocl_fn.init
    print ocl_fn.code

    print '*' * 5 + 'AsArray' + '*' * 50
    def function(y):
        return np.asarray([y[0], y[1], 3])

    ocl_fn = OCL_Function(function)
    print ocl_fn.init
    print ocl_fn.code

    print '*' * 5 + 'IfExp' + '*' * 50
    def function(y):
        return 5 if y > 3 else 0

    ocl_fn = OCL_Function(function)
    print ocl_fn.init
    print ocl_fn.code

    print '*' * 5 + 'Sign' + '*' * 50
    def function(y):
        return np.sign(y)

    ocl_fn = OCL_Function(function)
    print ocl_fn.init
    print ocl_fn.code

    print '*' * 5 + 'Radians' + '*' * 50
    power = 2
    def function(y):
        return np.radians(y**power)

    ocl_fn = OCL_Function(function)
    print ocl_fn.init
    print ocl_fn.code

    print '*' * 5 + 'Boolop' + '*' * 50
    power = 3.2
    def function(y):
        if y > 3 and y < 5:
            return y**power
        else:
            return np.sign(y)

    ocl_fn = OCL_Function(function)
    print ocl_fn.init
    print ocl_fn.code

    print '*' * 5 + 'Nested return' + '*' * 50
    power = 3.2
    def function(y):
        if y > 3 and y < 5:
            return y**power

        return np.sign(y)

    ocl_fn = OCL_Function(function)
    print ocl_fn.init
    print ocl_fn.code

    print '*' * 5 + 'Math constants' + '*' * 50
    def function(y):
        return np.sin(np.pi * y) + np.e

    ocl_fn = OCL_Function(function)
    print ocl_fn.init
    print ocl_fn.code
