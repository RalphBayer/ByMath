from __future__ import division
import math

from . import t_funcs
from . import tensor_object as t

e = math.e


def add(vals):
    val1, val2 = vals
    return val1 + val2

def sub(vals):
    val1, val2 = vals
    print(val1, val2)
    return val1 - val2

def mul(vals):
    val1, val2 = vals
    return val1 * val2

def truediv(vals):
    val1, val2 = vals
    return val1 / val2

def floordiv(vals):
    val1, val2 = vals
    return val1 // val2

def pow(vals):
    val1, val2 = vals
    return val1 ** val2


def neg(vals):
    val = vals[0]
    return -val

def pos(vals):
    val = vals[0]
    return +val


def exp(vals):
    if isinstance(vals, t.Tensor):
        return t_funcs.map_to_func(exp, vals, shape=vals.shape)
    elif isinstance(vals, (int, float, bool)):
        vals =  [vals]

    val = vals[0]
    return math.exp(val)
