from itertools import product
from functools import reduce
from operator import getitem
import random
from uuid import uuid4

from joblib import Parallel, delayed
import numpy as np


class GridParallel(object):

    def __init__(self, *args, **kwargs):
        """Wrapper for joblib.Parallel for computing the same function over a grid of parameters. Each function call has
        both the python and numpy random seeds set to a different random value.

        :param args: args for joblib.Parallel
        :param kwargs: kwargs for joblib.Parallel
        """

        self.parallel = Parallel(*args, **kwargs)

    def __call__(self, fun, *args):
        """Run a parallel job over a grid, calling the same function with the "coordinates" of the grid as arguments.
        Returns the grid of returned function values. The "coordinates" of the grid are provided as the additional
        arguments (*args), with each argument being the values of one axis. For example, if this is invoked with
        __call__(lambda x, y: (x, y), range(10), ['a', 'b']), then the output will be a 10 x 2 list of lists where each
        element is a tuple such as (7, 'b').

        :param fun: function to run at each point on the grid
        :param args: coordinate axes to construct the grid
        :return: list of lists of lists of ... of size S1 x S2 x S3 x ... where S1, S2, ... are the lengths of the
        coordinate axes provided in *args
        """

        grid_shape = tuple(len(x) for x in args)
        grid = self.build_grid(grid_shape)
        print('Number of tasks: %d.' % reduce(lambda x, y: x*y, grid_shape, 1))

        fs = (delayed(WrapFunctionRandomSeed(WrapFunctionWithReturn(fun, indices)))(*coordinates)
              for indices, coordinates in self.indices_coordinates_generator(*args))

        for val, indices in self.parallel(fs):
            self.assign_to_grid(grid, indices, val)

        return grid

    @staticmethod
    def build_grid(shape):

        if isinstance(shape, int):
            return [None]*shape
        elif len(shape) == 1:
            return [None]*shape[0]
        else:
            return [GridParallel.build_grid(shape[1:]) for _ in range(shape[0])]

    @staticmethod
    def assign_to_grid(grid, indices, val):

        if isinstance(indices, int):
            grid[indices] = val
        else:
            reduce(getitem, indices[:-1], grid)[indices[-1]] = val

    @staticmethod
    def indices_coordinates_generator(*args):

        for index_coordinate_pairs in product(*(enumerate(x) for x in args)):
            indices, coordinates = zip(*index_coordinate_pairs)
            yield indices, coordinates


class FakeParallel(object):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, iterable):
        res = []
        for f, *a in iter(iterable):
            if len(a) >= 1:
                args = a[0]
            if len(a) >= 2:
                kwargs = a[1]
            res.append(f(*args, **kwargs))
        return res


class WrapFunctionRandomSeed(object):

    def __init__(self, fun):
        self.fun = fun
        # create a random 32-bit unsigned integer to use as the seed
        self.seed = uuid4().int % 0x100000000

    def __call__(self, *args, **kwargs):
        random.seed(self.seed)
        np.random.seed(self.seed)
        return self.fun(*args, **kwargs)


class WrapFunctionWithReturn(object):

    def __init__(self, fun, val):
        self.fun = fun
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs), self.val
