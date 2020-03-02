from ..tensor import tensor_object as t

HAS_NUMPY  = False
HAS_SCIPY  = False
HAS_PANDAS = False

# TODO: ADD ANY MORE MAJOR TENSOR OR ARRAY HANDELING LIBARIES

def has_major_tensor_libs():
    """
        Checks if the user has certain major array
        and tensor handeling libaries installed.

    """

    global HAS_NUMPY, HAS_SCIPY, HAS_PANDAS

    # Check for numpy
    try:
        import numpy
        HAS_NUMPY = True
    except ImportError:
        HAS_NUMPY = False

    # Check for scipy
    try:
        import scipy
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    # Check for pandas
    try:
        import pandas
        HAS_PANDAS = True
    except ImportError:
        HAS_PANDAS = False

    # TODO: TRY TO MAKE THIS SHORTER?


def to_tensor_object(arg_pos):
    def decorator(func):
        def wrapper(*args, **kwargs):
            has_major_tensor_libs()
            new_args   = []
            new_kwargs = {}

            if isinstance(arg_pos, range):
                if arg_pos.start == arg_pos.stop:
                    arg_range = range(arg_pos.start, len(args))
            else:
                arg_range = arg_pos

            # TODO: ALSO GO THROUGH kwargs
            for i in range(len(args)):
                if i in arg_range:
                    # TODO: ADD IF STATEMENT FOR REST OF MAJOR LIBARIES
                    if HAS_NUMPY:
                        import numpy
                        if isinstance(args[i], numpy.ndarray):
                            new_args.append(t.Tensor(args[i].tolist()))

                    elif isinstance(args[i], (int, float, bool, list)):
                        new_args.append(t.Tensor(args[i]))

                    elif isinstance(args[i], t.Tensor):
                        new_args.append(args[i])

                    else:
                        raise ValueError(f"Can't turn var of type {type(args[i])} to Tensor object")
                else:
                    new_args.append(args[i])

            return func(*new_args, **kwargs)
        return wrapper
    return decorator


def tensors_to_same_shape(arg_pos):
    def decorator(func):
        @to_tensor_object(arg_pos)
        def wrapper(*args, **kwargs):
            biggest_shape = []

            if isinstance(arg_pos, range):
                if arg_pos.start == arg_pos.stop:
                    arg_range = range(arg_pos.start, len(args))
            else:
                arg_range = arg_pos

            # Get the shape all others will be reshaped to.
            # TODO: ALSO GO THROUGH kwargs
            for i in range(len(args)):
                if i in arg_range:
                    if len(args[i].shape) > len(biggest_shape):
                        biggest_shape = args[i].shape
                    else:
                        continue

            # Change the shape of each tensor
            for i in range(len(args)):
                if i in arg_range:
                    args[i].match_shape(biggest_shape)

            return func(*args, **kwargs)
        return wrapper
    return decorator
