class Index(object):
    def __init__(self, t_shape):
        """
            This class serves as a helper for the Tensor object.
            It keeps track of the tensor's index. It also contains
            many helpful functions for the Tensor object.

        """

        self.shape   = t_shape
        self.t_index = self.reset_index() # index in tensor form.





    def reset_index(self):
        """
            Resets the tensor's index all to zeros.

        """

        return [0] * len(self.shape.shape)


    def increment(self, value=1, index=-1):
        """
            Increments the index by 'value' at 'index'

        """

        self.t_index[index] += value
        self.check_for_indexerror()


    def check_for_indexerror(self):
        """
            Makes sure that when index is used, it doesn't cause
            an IndexError.

        """

        for i in range(len(self.t_index)-1, -1, -1): # Go backwards.
            if self.t_index[i] >= self.shape[i]:
                self.t_index[i] = 0

                if i-1 != -1:
                    self.t_index[i-1] += 1


    def as_array_index(self):
        """
            Turns the tensor index into array format.
            Example:
                # Tensor with shape [2, 2, 2]
                # [0, 1, 1] -> [3]

        """

        array_index = 0

        for i in range(len(self.t_index)):
            array_index += self.shape.get_size_from_dims(i+1) * self.t_index[i]

        return array_index


    def tslices_to_aslice(self, tslices):
        """
            Converts slices in tensor format to slices in array format.

        """

        if isinstance(tslices, tuple):
            tslices = list(tslices)
        else:
            tslices = [tslices]


        aslice = [0, 0, 0]
        for i in range(len(tslices)):

            if isinstance(tslices[i], int):
                tslices[i] = slice(tslices[i], tslices[i]+1, None)


            start = tslices[i].start
            stop  = tslices[i].stop
            step  = tslices[i].step

            # Slices to ints.
            if start == None:
                start = 0

            if stop == None:
                stop = self.shape[i]

            if step == None:
                step = 1


            # Calculate aslice.
            try:
                aslice[0] += start * self.shape.get_size_from_dims(i+1)
                aslice[1] += stop  * self.shape.get_size_from_dims(i+1)
                aslice[2] =  step

            except IndexError:
                raise TypeError("Can't take slice of bottom dimension from tensor.")


        return slice(aslice[0], aslice[1], aslice[2])


# TODO:
#   FIX tslices_to_aslice FUNCTION TO ALLOW FOR MORE THAN 1 SLICE BEING PASSED THROUGH. i.e tensor[0:2, 0::2]
#   ALSO TRY TO CLEAN UP ANY CODE.
