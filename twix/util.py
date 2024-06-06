import struct, re
from typing import Any, Dict

import six


def _read_cstr(source_file):
    '''Read a null terminated byte string'''
    chars = []
    char = source_file.read(1)
    end_char = six.b('\x00')
    while char != end_char:
        chars.append(char)
        char = source_file.read(1)

    out = six.b('').join(chars)
    if six.PY3:
        out = out.decode()
    return out 


class InvalidElemFormat(Exception):
    pass


def get_elem_mult(elem_fmt: str) -> int:
    mtch = re.match(r'([0-9]*)([cbB?hHiIlLqQnNefdsp])', elem_fmt)
    if not mtch:
        raise InvalidElemFormat(f"Not a valid format for element: {elem_fmt}")
    count, fmt = mtch.groups()
    if count != '' and fmt[-1] not in ('s', 'p'):
        return int(count)
    return 1


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def copy(self) -> "dotdict":
        return dotdict(super().copy())


class BinaryHeader:
    """Allow reading and writing binary headers based on a simple `spec` dict"""
    def __init__(self, spec: Dict[str, str], order_size_align: str = '<'):
        self._spec = spec.copy()
        self._fmt = order_size_align + ''.join(self._spec.values())
        self._size = struct.calcsize(self._fmt)
        self._elem_mults = {n: get_elem_mult(f) for n, f in self._spec.items()}

    @property
    def fmt(self) -> str:
        return self._fmt

    @property
    def size(self) -> int:
        return self._size
    
    def read(self, in_file) -> dotdict:
        elems = struct.unpack(self._fmt, in_file.read(self._size))
        res = dotdict()
        curr_idx = 0
        for name in self._spec:
            mult = self._elem_mults[name]
            if mult == 1:
                res[name] = elems[curr_idx]
            else:
                res[name] = tuple(elems[curr_idx:curr_idx+mult])
            curr_idx += mult
        return res

    def write(self, data, out_file) -> int:
        flat = []
        for sub_val in data.values():
            if hasattr(sub_val, '__iter__') and not isinstance(sub_val, (str, bytes)):
                flat.extend(sub_val)
            else:
                flat.append(sub_val)
        return out_file.write(struct.pack(self._fmt, *flat))
    
    def __iter__(self):
        for name in self._spec:
            yield name
