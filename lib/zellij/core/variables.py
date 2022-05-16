from abc import ABC, abstractmethod
import numpy as np
import random

import logging

logger = logging.getLogger("zellij.variables")
logger.setLevel(logging.INFO)


class Solution(ABC):
    pass


@abstractmethod
class Variable(ABC):
    def __init__(self, label, condition=None):
        assert isinstance(label, str), logger.error(
            f"Label must be a string, got {label}"
        )
        self.label = label

    @abstractmethod
    def random(self, size=None):
        pass

    @abstractmethod
    def isconstant(self):
        pass


# Discrete
class IntVar(Variable):
    def __init__(self, label, lower, upper, condition=None):
        super(IntVar, self).__init__(label, condition=None)

        assert isinstance(upper, int), logger.error(
            f"Upper bound must be an int, got {upper}"
        )

        assert isinstance(lower, int), logger.error(
            f"Upper bound must be an int, got {lower}"
        )

        assert lower < upper, logger.error(
            f"Lower bound must be strictly inferior to upper bound,\
            got {lower}<{upper}"
        )

        self.up_bound = upper
        self.low_bound = lower

    def random(self, size=None):
        return np.random.randint(self.low_bound, self.up_bound, size, dtype=int)

    def isconstant(self):
        return self.up_bound == self.lo_bounds


# Real
class FloatVar(Variable):
    def __init__(
        self, label, lower, upper, sampler=np.random.uniform, condition=None
    ):
        super(FloatVar, self).__init__(label, condition=None)

        assert isinstance(upper, float) or isinstance(upper, int), logger.error(
            f"Upper bound must be an int or a float, got {upper}"
        )

        assert isinstance(lower, int) or isinstance(lower, float), logger.error(
            f"Upper bound must be an int or a float, got {lower}"
        )

        assert lower < upper, logger.error(
            f"Lower bound must be strictly inferior to upper bound, got {lower}<{upper}"
        )

        self.up_bound = upper
        self.low_bound = lower
        self.sampler = sampler

    def random(self, size=None):
        return self.sampler(self.low_bound, self.up_bound, size)

    def isconstant(self):
        return self.up_bound == self.lo_bounds


# Categorical
class CatVar(Variable):
    def __init__(self, label, features, weights=None, condition=None):
        super(CatVar, self).__init__(label, condition=None)

        assert isinstance(features, list), logger.error(
            f"Features must be a list with a length > 0, got{features}"
        )

        assert len(features) > 0, logger.error(
            f"Features must be a list with a length > 0,\
             got length= {len(features)}"
        )

        self.features = features

        assert isinstance(weights, list) or weights == None, logger.error(
            f"`weights` must be a list or equal to None, got {weights}"
        )

        if weights:
            self.weights = weights
        else:
            self.weights = [1 / len(features)] * len(features)

    def random(self, size=1):
        if size == 1:
            res = random.choices(self.features, weights=self.weights, k=size)[0]
            if isinstance(res, Variable):
                res = res.random()
        else:
            res = random.choices(self.features, weights=self.weights, k=size)

            for v in res:
                if isinstance(res, Variable):
                    v = v.random()

        return res

    def isconstant(self):
        return len(self.features) == 1


# Array of variables
class ArrayVar(Variable):
    def __init__(self, label, *args, condition=None):
        super(ArrayVar, self).__init__(label, condition=None)

        assert all(isinstance(v, Variable) for v in args), logger.error(
            f"All elements must inherit from `Variable`, got {args}"
        )
        self.values = args

    def random(self, size=1):

        if size == 1:
            return [v.random() for v in self.values]
        else:
            res = []
            for _ in range(size):
                res.append([v.random() for v in self.values])

            return res

    def isconstant(self):
        return all(v.isconstant for v in self.values)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):

        if self.index >= len(self.values):
            raise StopIteration

        res = self.values[self.index]
        self.index += 1
        return res


# Block of variable, fixed size
class Block(Variable):
    def __init__(self, label, value, repeat, condition=None):
        super(Block, self).__init__(label, condition)

        assert isinstance(value, Variable), logger.error(
            f"Value must inherit from `Variable`, got {args}"
        )
        self.value = value

        assert isinstance(repeat, int) and repeat > 0, logger.error(
            f"`repeat` must be a strictly positive int, got {repeat}."
        )
        self.repeat = repeat

    def random(self, size=1):
        res = []

        if size > 1:
            for _ in range(size):
                block = []
                for _ in range(self.repeat):
                    block.append([v.random() for v in self.value])
                res.append(block)
        else:
            for _ in range(self.repeat):
                res.append([v.random() for v in self.value])

        return res

    def isconstant(self):
        return all(v.isconstant() for v in self.value)


# Block of variables, with random size.
class DynamicBlock(Block):
    def __init__(self, label, value, repeat, condition=None):
        super(DynamicBlock, self).__init__(label, value, repeat, condition)

    def random(self, size=1):
        res = []
        n_repeat = np.random.randint(1, self.repeat)

        if size > 1:
            for _ in range(size):
                block = []
                for _ in range(n_repeat):
                    block.append([v.random() for v in self.value])
                res.append(block)
        else:
            for _ in range(n_repeat):
                res.append([v.random() for v in self.value])

        return res

    def isconstant(self):
        return False
