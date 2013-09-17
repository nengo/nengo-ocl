
import inspect, ast, collections
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

function_map = {
    np.abs: 'fabs',
    np.absolute: 'fabs',
    np.add: lambda args: args[0] + " + " + args[1],
    np.arccos: 'acos',
    np.arccosh: 'acosh',
    np.arcsin: 'asin',
    np.arcsinh: 'asinh',
    np.arctan: 'atan',
    np.arctan2: 'atan2',
    np.arctanh: 'atanh',
    np.bitwise_and: lambda args: args[0] + " & " + args[1],
    np.bitwise_not: lambda args: "~" + args[0],
    np.bitwise_or: lambda args: args[0] + " | " + args[1],
    np.bitwise_xor: lambda args: args[0] + " ^ " + args[1],
    np.ceil: 'ceil',
    np.copysign: 'copysign',
    np.cos: 'cos',
    np.cosh: 'cosh',
    np.deg2rad: lambda args: args[0] + " * M_PI / 180",
    np.degrees: lambda args: args[0] + " * 180 * M_1_PI",
    np.divide: lambda args: args[0] + " / " + args[1],
    np.equal: lambda args: args[0] + " == " + args[1],
    np.exp: 'exp',
    np.exp2: 'exp2',
    np.expm1: 'expm1',
    np.fabs: 'fabs',
    np.floor: 'floor',
    np.floor_divide: lambda args: "floor(" + args[0] + " / " + args[1] + ")",
    np.fmax: 'fmax',
    np.fmin: 'fmin',
    np.fmod: 'fmod',
    np.greater: lambda args: args[0] + " > " + args[1],
    np.greater_equal: lambda args: args[0] + " >= " + args[1],
    np.hypot: "hypot",
    np.invert: lambda args: "~" + args[0],
    np.isfinite: 'isfinite',
    np.isinf: 'isinf',
    np.isnan: 'isnan',
    np.ldexp: 'ldexp',
    np.left_shift: lambda args: "<<" + args[0],
    np.less: lambda args: args[0] + " < " + args[1],
    np.less_equal: lambda args: args[0] + " <= " + args[1],
    np.log: 'log',
    np.log10: 'log10',
    np.log1p: 'log1p',
    np.log2: 'log2',
    np.logaddexp: lambda args: "log(exp(" + args[0] + "), exp(" + args[1] + "))",
    np.logaddexp: lambda args: "log2(exp2(" + args[0] + "), exp2(" + args[1] + "))",
    np.logical_and: lambda args: args[0] + " && " + args[1],
    np.logical_not: lambda args: "!" + args[0],
    np.logical_or: lambda args: args[0] + " || " + args[1],
    np.logical_xor: lambda args: args[0] + " ^^ " + args[1],
    np.maximum: 'fmax',
    np.minimum: 'fmin',
    np.mod: 'remainder',
    np.modf: lambda args: "floor(" + args[0] + ")" if len(args)==1 else 'modf(' + args[0] + ", *" + args[2] + ")",
    np.multiply: lambda args: args[0] + " * " + args[1],
    np.negative: lambda args: "-" + args[0],
    np.nextafter: 'nextafter',
    np.power: 'pow',
    np.prod: lambda args: args[0] + " * " + args[1],
    np.product: lambda args: args[0] + " * " + args[1],
    np.rad2deg: lambda args: args[0] + " * 180 * M_1_PI",
    np.radians: lambda args: args[0] + " * M_PI / 180",
    np.reciprocal: lambda args: "1 / " + args[0],
    np.remainder: 'remainder',
    np.rint: 'rint',
    np.sign: lambda args: args[0] + "==0 ? 0 : " + args[0] + ">0 ? 1 : 0",
    np.signbit: lambda args: args[0] + " < 0",
    np.sin: 'sin',
    np.sinh: 'sinh',
    np.sqrt: 'sqrt',
    np.square: lambda args: args[0] + " * " + args[0],
    np.subtract: lambda args: args[0] + " - " + args[1],
    np.tan: 'tan',
    np.tanh: 'tanh',
    math.acos: 'acos',
    math.acosh: 'acosh',
    math.asin: 'asin',
    math.asinh: 'asinh',
    math.atan: 'atan',
    math.atan2: 'atan2',
    math.atanh: 'atanh',
    math.ceil: 'ceil',
    math.copysign: 'copysign',
    math.cos: 'cos',
    math.cosh: 'cosh',
    math.degrees: lambda args: args[0] + " * 180 * M_1_PI",
    math.erf: 'erf',
    math.erfc: 'erfc',
    math.exp: 'exp',
    math.expm1: 'expm1',
    math.fabs: 'fabs',
    math.floor: 'floor',
    math.fmod: 'fmod',
    math.gamma: lambda args: 'exp(lgamma(' + args[0] + '))',
    math.hypot: 'hypot',
    math.isinf: 'isinf',
    math.isnan: 'isnan',
    math.ldexp: 'ldexp',
    math.lgamma: 'lgamma',
    math.log: 'log',
    math.log10: 'log10',
    math.log1p: 'log1p',
    math.pow: 'pow',
    math.sin: 'sin',
    math.sinh: 'sinh',
    math.sqrt: 'sqrt',
    math.tan: 'tan',
    math.tanh: 'tanh',
    }

INPUT_NAME = "__INPUT__"
OUTPUT_NAME = "__OUTPUT__"

class OCL_Translator(ast.NodeVisitor):
    def __init__(self, source, globals_dict, closure_dict, filename=None):
        self.source = source
        self.globals = globals_dict
        self.closures = closure_dict
        self.filename = filename

        ### parse and make code
        a = ast.parse(source)
        function_def = a.body[0]

        self.arg_names = [arg.id for arg in function_def.args.args]

        self.filename = filename
        self.function_name = function_def.name

        self.init = collections.OrderedDict()
        self.body = self.visit_block(function_def.body)

    def _var_to_string(self, var):
        if isinstance(var, str):
            return '"%s"' % var
        elif isinstance(var, float):
            ### TODO: can we get around putting the 'f' afterwards?
            ### Append an 'f' to floats, o.w. some calls (e.g. pow) ambiguous
            return "%sf" % var
        elif isinstance(var, int):
            return str(var)
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
            return '%s[0]' % name
        elif name in self.closures:
            return self._var_to_string(self.closures[name])
        elif name in self.globals:
            return self._var_to_string(self.globals[name])
        elif name in self.init:
            return name
        else:
            raise ValueError("Unrecognized name '%s'" % name)

    def visit_Num(self, expr):
        return self._var_to_string(expr.n)

    def visit_Str(self, expr):
        return self._var_to_string(expr.s)

    def visit_Index(self, expr):
        return self.visit(expr.value)

    def visit_Ellipsis(self, expr):
        raise NotImplementedError("Ellipsis")

    def visit_Slice(self, expr):
        raise NotImplementedError("Slice")

    def visit_ExtSlice(self, expr):
        raise NotImplementedError("ExtSlice")

    def _needs_wrapping(self, expr):
        if isinstance(expr, (ast.Num, ast.Name, ast.Call, ast.Subscript)):
            return False
        elif isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Pow):
            return False
        else:
            return True

    def _visit_binary_op(self, op, left, right):
        s_left = self.visit(left)
        s_right = self.visit(right)
        if self._needs_wrapping(left): s_left = "(%s)" % s_left
        if self._needs_wrapping(right): s_right = "(%s)" % s_right

        opt = type(op)
        if opt in infix_binary_ops:
            return "%s %s %s" % (s_left, infix_binary_ops[opt], s_right)
        elif opt is ast.Pow:
            if isinstance(right, ast.Num):
                if isinstance(right.n, int):
                    return "pown(%s, %s)" % (s_left, s_right)
                elif right.n > 0:
                    return "powr(%s, %s)" % (s_left, s_right)
            return "pow(%s, %s)" % (s_left, s_right)
        else:
            raise NotImplementedError(
                "'%s' operator is not supported" % opt.__name__)

    def visit_UnaryOp(self, expr):
        value = self.visit(expr.operand)
        opt = type(expr.op)
        if opt in prefix_unary_ops:
            return "%s%s" % (prefix_unary_ops[opt], value)
        else:
            raise NotImplementedError(
                "'%s' operator is not supported" % opt.__name__)

    def visit_BinOp(self, expr):
        return self._visit_binary_op(expr.op, expr.left, expr.right)

    def visit_BoolOp(self, expr):
        raise NotImplementedError("BoolOp")

    def visit_Compare(self, expr):
        assert len(expr.ops) == 1
        assert len(expr.comparators) == 1
        return self._visit_binary_op(
            expr.ops[0], expr.left, expr.comparators[0])

    def visit_Subscript(self, expr):
        assert isinstance(expr.value, ast.Name)
        assert isinstance(expr.slice, ast.Index), "Slicing is not supported"
        # var = self.visit(expr.value)
        index = self.visit(expr.slice)
        return "%s[%s]" % (expr.value.id, index)

    def visit_Call(self, expr):
        assert expr.kwargs is None, "kwargs not implemented"

        def get_handle(expr):
            if isinstance(expr, ast.Name):
                return (self.closures[expr.id] if expr.id in self.closures
                        else self.globals[expr.id])
            else:
                return getattr(get_handle(expr.value), expr.attr)
        handle = get_handle(expr.func)

        if handle in function_map:
            value = function_map[handle]
            args = [self.visit(arg) for arg in expr.args]
            if callable(value):
                return "(%s)" % value(args)
            else:
                return "%s(%s)" % (value, ', '.join(args))
        else:
            raise NotImplementedError(
                "'%s' function is not supported" % handle.__name__)

    def visit_List(self, expr):
        raise NotImplementedError("List")

    def visit_Expr(self, expr):
        raise NotImplementedError("Expr")

    def visit_GeneratorExp(self, expr):
        raise NotImplementedError("GeneratorExp")

    def visit_ListComp(self, expr):
        raise NotImplementedError("ListComp")

    def visit_Attribute(self, expr):
        raise NotImplementedError("Attribute")

    def visit_Tuple(self, expr):
        raise NotImplementedError("Tuple")

    def visit_IfExp(self, expr):
        cond = self.visit(expr.test)
        true = self.visit(expr.body)
        false = self.visit(expr.orelse)
        return "%s ? %s : %s" % (cond, true, false)

    def visit_Print(self, expr):
        assert expr.dest is None, "other dests not implemented"
        if (len(expr.values) == 1
            and isinstance(expr.values[0], ast.BinOp)
            and isinstance(expr.values[0].op, ast.Mod)
            and isinstance(expr.values[0].left, ast.Str)):
            # we're using string formatting
            stmt = self.visit(expr.values[0].left)[:-1] + '\\n"'
            if isinstance(expr.values[0].right, ast.Tuple):
                args = [self.visit(arg) for arg in expr.values[0].right.elts]
            else:
                args = [self.visit(expr.values[0].right)]
            return ["printf(%s);" % ', '.join([stmt] + args)]
        else:
            stmt = '"' + ' '.join(['%s' for arg in expr.values]) + '\\n"'
            args = ', '.join([self.visit(arg) for arg in expr.values])
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
        rhs = self.visit(expr.value)
        lhs = self.visit_lhs(expr.targets[0])
        return ["%s = %s;" % (lhs, rhs)]

    def visit_AugAssign(self, expr):
        new_value = self.visit_lhs(expr.target)
        target = self._visit_binary_op(expr.op, expr.target, expr.value)
        return ["%s = %s;" % (new_value, target)]

    def visit_Return(self, expr):
        if isinstance(expr.value, ast.List):
            return ["%s[%d] = %s;" % (OUTPUT_NAME, i, self.visit(elt))
                    for i, elt in enumerate(expr.value.elts)]
        else:
            return ["%s[0] = %s;" % (OUTPUT_NAME, self.visit(expr.value))]

    def visit_If(self, expr):
        return ["if (%s) {" % self.visit(expr.test),
                self.visit_block(expr.body),
                "} else {",
                self.visit_block(expr.orelse),
                "}"]

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
        if self._is_lambda(self.fn):
            raise NotImplementedError("No lambda functions")
        elif self.fn in function_map:
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
                lines.extend(flatten(b, indent+4))
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
    # @OCL_Function
    # def square(x):
    #     print "wow: %f, %d, %s" % (0.3, 9, "hello")
    #     if x < 0.5 - 0.1:
    #         x = 2. * x
    #         x -= 4 + (3 if x > 99 else 2)
    #     elif x == 2:
    #         x *= 9.12
    #         x = 4*(x - 2)
    #     else:
    #         x = 9*x
    #         x += x**(-1.1)

    #     return np.sin(multiplier * (x * x))

    # print square(4)
    # print square.get_ocl_code()

    # multiplier = 3.14

    # def slicing(x):
    #     y = x**3
    #     return x[0] * x[1]
    #     # return [x * x, multiplier*y]

    # function = np.sin

    def function(dede):
        return np.subtract(dede, dede)

    ocl_fn = OCL_Function(function)
    print ocl_fn.init
    print ocl_fn.code


