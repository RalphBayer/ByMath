import bymath as bm

t1 = bm.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], shape=[2, 2, 2])

# Sigmoid function
print(1/(1+bm.exp(-t1)))

# TODO:
#   IMPROVE tslices_to_aslice FUNCTION IN TENSOR CLASS.
#   FINISH ADDING MATH FUNCTIONS FOR TENSOR OBJECT.
#   ADD A tensor_print FUNCTION.
#   ADD DATA TYPES.
#   ADD INITIALIZATION FUNCTIONS FOR THE TENSOR CLASS.
#   CYTHONIZE EVERYTHING
