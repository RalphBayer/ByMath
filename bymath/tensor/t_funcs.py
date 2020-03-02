import re

from . import tensor_object as t
from ..utils.t_decorators import *

@to_tensor_object(0)
def as_tensor(py_object, shape=[]):
    """
        Takes a python object and converts it to a tensor_object.

    """

    if shape != []:
        py_object._shape.shape = shape
        py_object._shape.rank  = len(shape)
        py_object.reset_index()

    return py_object


def flatten(py_array):
    """
        Takes a python array and flattens it's 'dimensions'.

    """

    flattened = re.sub(r'[\[,\]]', '', str(py_array)).split()
    return [float(i) for i in flattened]


@tensors_to_same_shape(range(1,1))
def map_to_func(func, *args, shape=[]):
    """
        Maps a python array or Tensor object to a function.
        The function should accept single values for each py_array.
        All py_arrays are streched to math the shape of each other.
        The function should also accept the values all together in a list.

        Example:
            >>>  def add(vals):
            >>>     val1, val2 = vals
            >>>     return val1 + val2
            >>>
            >>> bm.map_to_func(add, tensor1, tensor2)

    """

    new_array = []
    for _ in range(args[0].size):

        # get each values
        vals = []

        for arg in args:
            vals.append(arg.at_current_index())

        new_array.append(func(vals))

        # increment each py_arrays' index
        for arg in args:
            arg.increment_index()

    return as_tensor(new_array, shape)
