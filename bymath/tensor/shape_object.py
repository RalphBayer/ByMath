import copy
from functools import reduce

from . import t_funcs

class Shape(object):
    def __init__(self, py_array):
        """
            This class serves as a helper for the Tensor object.
            It keeps track of the tensor shape. It also contains
            many helpful functions for the Tensor object.

        """

        self.size  = self.get_size(py_array)
        self.shape = self.get_shape(py_array)
        self.rank  = self.get_rank()

        # Check for different sizes in same dimesions. Can't have [[1, 2, 3], [4, 5]]
        if self.get_size_from_dims(0) != self.size:
            raise TypeError("Every size of each array in same dimesion must match.")





    def get_size(self, py_array):
        """
            Calculates the number of items in py_array.

        """

        return len(t_funcs.flatten(py_array))


    def get_shape(self, py_array):
        """
            Calculates the shape (size of each dimesion) of a python array.

        """

        # Is py_array a scalar.
        if isinstance(py_array, (int, float, bool)):
            return [1]

        shape = []
        while True:
            # Try to find a list, which indicates a new dimension.
            if isinstance(py_array, list):
                # Repeat dimension checking and add dimension to shape.
                shape.append(len(py_array))
                py_array = py_array[0]

            else:
                return shape


    def get_rank(self):
        """
            Calculates the number of dimesions

        """

        return len(self.shape)


    def get_size_from_dims(self, index):
        """
            Calculates the number of items combined from a certain dim to the rest.

        """

        size = 1

        for i in range(self.rank - index):
            size *= self.shape[i + index]

        return size


    def reshape(self, value, index=slice(0, None, None)):
        """
            Tries to reshape the tensor object's shape
            while keeping size the same.

        """

        temp_shape = copy.copy(self.shape)
        temp_shape[index] = value

        if reduce((lambda x, y: x*y), temp_shape) != self.size:
            raise TypeError(f"Cant reshape Tensor with shape {self.shape} to {temp_shape}")

        self.shape[index] = value

    def add_dim(self, index):
        """
            Adds a dim of size 1 to self.shape at index.

        """

        self.shape.insert(index, 1)

    def match_shape(self, shape_to_match):
        """
            Reshapes self.shape to be compatible with shape_to_match
        """

        for i in range(len(shape_to_match)-1, -1, -1):

            if len(shape_to_match[i:]) == len(self.shape):
                for j in range(len(self.shape)-1, -1, -1):

                    if self.shape[j] != 1 and shape_to_match[i:][j] != 1:
                       if self.shape[j] != shape_to_match[i:][j]:
                           raise ValueError(f"Can't match shape {self.shape} to {shape_to_match}")

                    if len(self.shape) != len(shape_to_match):
                        self.add_dim(0)





    # Dunder or 'magic' functions
    def __getitem__(self, index):
        if isinstance(index, slice):
            if index.stop != None or index.start != None:
                if index.stop > self.rank or index.start > self.rank:
                    raise IndexError("Shape object does not have enough dimesions.")

        return self.shape[index]


    def __setitem__(self, index, value):
        if isinstance(index, slice):
            if index.stop != None or index.start != None:
                if index.stop > self.rank or index.start > self.rank:
                    raise IndexError("Shape object does not have enough dimesions.")

        self.reshape(value, index)


    def __str__(self):
        return str(self.shape)
