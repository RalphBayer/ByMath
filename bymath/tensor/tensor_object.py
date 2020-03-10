 import Index
from .shape_object import Shape

from . import t_funcs
from . import t_maths


class Tensor(object):
    def __init__(self, py_array, shape=[], dtype="int64"):
        """
            The Tensor class is a representation of a mathematical tensor.
            By keeping all values of the tensor in a single list and by
            using maths, the dimensions or "rank" of a tensor can be
            simulated.

            Example:
                >>> my_tensor = Tensor([[1, 2, 3], [4, 5, 6]])
                >>> # Stored as [1, 2, 3, 4, 5, 6]
                >>> # Represented as [[1, 2, 3], [4, 5, 6]]

            Arguments:
                tensor : list : A python list of values for the tensor.
                dtype  :      : The data type all values get casted to.

        """

        self._tensor = t_funcs.flatten(py_array) # TODO: CAST TO dtype AFTER flatten().
        self._shape  = Shape(py_array)

        if shape != []:
            self.shape = shape

        self._index   = Index(self._shape)


    # getter and setter functions.
    @property
    def tensor(self): return self._tensor
    @tensor.setter
    def tensor(self, value):
        if isinstance(value, list):
            self.__init__(value)

    @property
    def shape(self): return self._shape.shape
    @shape.setter
    def shape(self, value): self._shape.reshape(value)

    @property
    def size(self): return self._shape.size

    @property
    def rank(self): return self._shape.rank





    # Functions for the tensor's shape object
    def flatten(self):
        """
            Flattens the shape of the tensor.

        """

        self._shape.shape = self._shape.get_size_from_dims(0)


    def reshape(self, value, index=slice(0, None, None)):
        """
            Reshapes the tensor's shape.

        """

        self._shape.reshape(value, index)


    def add_dim(self, index):
        """
            Adds a dim of size 1 to self.shape at index.

        """
        self._shape.add_dim(index)


    def match_shape(self, shape_to_match):
        """
            Reshapes self.shape to be compatible with shape_to_match

        """

        self._shape.match_shape(shape_to_match)




    # Functions for the tensor's index object
    def value_at_current_index(self):
        """
            Returns a list of the tensor at self._index.index.

        """

        # Should I cast it to a tensor object here or leave it as a list?
        return self.tensor[self._index.as_array_index()]


    def increment_index(self, value=1, index=-1):
        """
            adds 'value' to self._index.index at 'index'

        """

        self._index.increment(value, index)


    def reset_index(self):
        """
            Turns self._index.index into a list of 0s with size len(self.shape)

        """

        self._index.reset_index()



    # Dunder or 'magic' functions:
    def __getitem__(self, tslices):
        aslice     = self._index.tslices_to_aslice(tslices)
        new_tensor = t_funcs.as_tensor(self.tensor[aslice], shape=self.shape[len([tslices]):])
        return new_tensor


    def __setitem__(self, tslice, value):
        aslice              = self._index.tslices_to_aslice(tslices)
        self.tensor[aslice] = value


    def __str__(self):
        return str(self.tensor)


    def __add__(self, add_tensor):
        return t_funcs.map_to_func(t_maths.add, self, add_tensor, shape=self.shape)

    def __radd__(self, add_tensor):
        return t_funcs.map_to_func(t_maths.add, self, add_tensor, shape=self.shape)

    def __iadd__(self, add_tensor):
        return t_funcs.map_to_func(t_maths.add, self, add_tensor, shape=self.shape)


    def __sub__(self, sub_tensor):
        return t_funcs.map_to_func(t_maths.sub, self, sub_tensor, shape=self.shape)

    def __rsub__(self, sub_tensor):
        return t_funcs.map_to_func(t_maths.sub, sub_tensor, self, shape=self.shape)

    def __isub__(self, sub_tensor):
        return t_funcs.map_to_func(t_maths.sub, self, sub_tensor, shape=self.shape)


    def __mul__(self, mul_tensor):
        return t_funcs.map_to_func(t_maths.mul, self, mul_tensor, shape=self.shape)

    def __rmul__(self, mul_tensor):
        return t_funcs.map_to_func(t_maths.mul, self, mul_tensor, shape=self.shape)

    def __imul__(self, mul_tensor):
        return t_funcs.map_to_func(t_maths.mul, self, mul_tensor, shape=self.shape)


    def __truediv__(self, div_tensor):
        return t_funcs.map_to_func(t_maths.truediv, self, div_tensor, shape=self.shape)

    def __rtruediv__(self, div_tensor):
        return t_funcs.map_to_func(t_maths.truediv, div_tensor, self, shape=self.shape)

    def __itruediv__(self, div_tensor):
        return t_funcs.map_to_func(t_maths.truediv, self, div_tensor, shape=self.shape)


    def __floordiv__(self, div_tensor):
        return t_funcs.map_to_func(t_maths.floordiv, self, div_tensor, shape=self.shape)

    def __rfloordiv__(self, div_tensor):
        return t_funcs.map_to_func(t_maths.floordiv, div_tensor, self, shape=self.shape)

    def __ifloordiv__(self, div_tensor):
        return t_funcs.map_to_func(t_maths.floordiv, self, div_tensor, shape=self.shape)


    def __pow__(self, pow_tensor):
        return t_funcs.map_to_func(t_maths.pow, self, pow_tensor, shape=self.shape)

    def __rpow__(self, pow_tensor):
        return t_funcs.map_to_func(t_maths.pow, pow_tensor, self, shape=self.shape)

    def __ipow__(self, pow_tensor):
        return t_funcs.map_to_func(t_maths.pow, self, pow_tensor, shape=self.shape)


    def __neg__(self):
        return t_funcs.map_to_func(t_maths.neg, self, shape=self.shape)

    def __pos__(self):
        return t_funcs.map_to_func(t_maths.pos, self, shape=self.shape)
