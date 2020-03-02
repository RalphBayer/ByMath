import bymath as bm

t1 = bm.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
t2 = 1

def add(vals):
    val1, val2 = vals
    return val1 + val2

t3 = bm.map_to_func(add, t1, t2, shape=[2, 2, 2])

# TODO:
#   IMPROVE tslices_to_aslice FUNCTION IN TENSOR CLASS.
#   ADD MATH FUNCTIONS FOR TENSOR OBJECT.
#   ADD A tensor_print FUNCTION.
#   ADD DATA TYPES.
#   ADD INITIALIZATION TYPES.
