from nengo.utils.compat import PY2


def as_ascii(string):
    if not PY2 and isinstance(string, bytes):  # Python 3
        return string.decode('ascii')
    elif not isinstance(string, str):
        return str(string)
    else:
        return string


def indent(s, i):
    return '\n'.join([(' ' * i) + line for line in s.split('\n')])
