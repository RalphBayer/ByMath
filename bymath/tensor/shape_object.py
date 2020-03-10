from . import t_funcs

class Shape(object):
    def __init__(self, py_array, shape=[]):
        """
            This class serves as a helper for the Tensor object.
            It keeps track of the tensor shape. It also contains
            many helpful functions for the Tensor object.

        """

        if shape == []:
            self.shape = self.get_shape(py_array)
        else:
            self.shape = shape

        self.size = self.get_size(py_array)
        self.rank = len(self.shape)

        # Check for different sizes in same dimesions.
        if self.get_size_from_dims(0) != self.size:
            raise ValueError("Every size of each array in the same dimesion must match.")
            # Can't have [[1, 2, 3], [4, 5]].




    # Functions that can be applied to any shape(s).
    def get_shape(self, py_array):
        """
            Finds the size of each dimension in a nested python array.

        """

        # Is py_array a scalar.
        if isinstance(py_array, (int, float, bool)):
            return [1]

        shape = []
        while True:
            # Executes if a nother dimension is found.
            if isinstance(py_array, (list, tuple)):
                shape.append(len(py_array))
                py_array = py_array[0]

            else:
                return shape


    def get_size(self, py_array):
        """
            Finds how many values are in the python nested list.

        """

        return len(t_funcs.flatten(py_array))


    def get_size_from_dims(self, index, shape=[]):
        """
            Finds the amount of values combined in each dimension from range(index, len(shape)).

        """

        if shape == []:
            shape = self.shape

        size = 1
        for i in range(index, len(shape), 1):
            size *= shape[i]

        return size


    def get_biggest_shape(self, shapes):
        """
            Finds the biggest shape in a list of shapes.

        """

        biggest_shape = [0]

        for shape in shapes:
            if len(shape) > len(biggest_shape):
                biggest_shape = shape

            elif len(shape) == len(biggest_shape) \
            and  self.get_size_from_dims(0, shape) > self.get_size_from_dims(0, biggest_shape):
                biggest_shape = shape

        return biggest_shape


    def get_biggest_size(self, sizes):
        """
            Finds the biggest size in a list of sizes.

        """

        biggest_size = 0

        for size in sizes:
            if size > biggest_size:
                biggest_size = size

        return biggest_size




    # Functions that get applies to self.shape.
    def reshape(self, value, index=slice(0, None, None)):
        """
            Tries to reshape the tensor object's shape
            while keeping self.size the same.

        """

        temp_shape = self.shape[0:]
        temp_shape[index] = value

        if self.get_size_from_dims(0, temp_shape) != self.size:
            raise ValueError(f"Cant reshape Tensor with shape {self.shape} to {temp_shape}")

        self.shape[index] = value
        self.rank = len(self.shape)


    def add_dim(self, indices):
        """
            Adds a dim to self.shape at indices.

        """

        if isinstance(indices, int):
            indices = [indices]

        for index in indices:
            self.shape.insert(index, 1)

        self.rank = len(self.shape)


    def match_shape(self, shape_to_match):
        """
            Rehsapes self.shape to be compatible with shape_to_match
            by adding new dimensions while keeping self.size the same.

        """

        # Go backwards through the sizes comparing each dimension.
        for i in range(len(shape_to_match)-1, -1, -1):
            if len(shape_to_match[i:]) == self.rank:
                for j in range(self.rank-1, -1, -1):

                    if self.shape[j] != 1 and shape_to_match[i:][j] != 1:
                       if self.shape[j] != shape_to_match[i:][j]:
                           raise ValueError(f"Can't match shape {self.shape} to {shape_to_match}")

                    if len(self.shape) != len(shape_to_match):
                        self.add_dim(0)





    # Dunder or "magic" functions.
    def __getitem__(self, index):
        if isinstance(index, slice):
            if index.stop != None or index.start != None:
                if index.stop > self.rank or index.start > self.rank:
                    raise IndexError("Shape object does not have enough dimesions.")
        elif index > self.rank:
            raise IndexError("Shape object does not have enough dimesions.")

        return self.shape[index]


    def __setitem__(self, index, value):
        if isinstance(index, slice):
            if index.stop != None or index.start != None:
                if index.stop > self.rank or index.start > self.rank:
                    raise IndexError("Shape object does not have enough dimesions.")
        elif index > self.rank:
            raise IndexError("Shape object does not have enough dimesions.")

        self.reshape(value, index)
