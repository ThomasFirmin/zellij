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
    """Variable

    `Variable` is an Abstract class defining what a variable is in a :ref:`sp`.

    Parameters
    ----------
    label : str
        Name of the variable.

    Attributes
    ----------
    label

    """

    def __init__(self, label):
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

    def __repr__(self):
        return f"\n{self.__class__.__name__}:\n\
        \t- Label: {self.label}\n"


# Discrete
class IntVar(Variable):
    """IntVar

    `IntVar` is a `Variable` discribing an Integer variable.

    Parameters
    ----------
    label : str
        Name of the variable.
    lower : int
        Lower bound of the variable
    upper : int
        Upper bound of the variable

    Attributes
    ----------
    up_bound : int
        Lower bound of the variable
    low_bound : int
        Upper bound of the variable

    Examples
    --------
    >>> from zellij.core.variables import IntVar
    >>> a = IntVar("test", 0, 5)
    >>> print(a)
    FloatVar:
                - Label: test
                - Lower bound: 0
                - Upper bound: 5.0
    >>> a.random()
    1

    """

    def __init__(self, label, lower, upper):
        super(IntVar, self).__init__(label)

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

        self.low_bound = lower
        self.up_bound = upper

    def random(self, size=None):
        """random(size=None)

        Parameters
        ----------
        size : int, default=None
            Number of draws.

        Returns
        -------
        out: int or list[int]
            Return an int if `size`=1, a list[int] else.

        """
        return np.random.randint(self.low_bound, self.up_bound, size, dtype=int)

    def isconstant(self):
        """isconstant()

        Returns
        -------
        out: boolean
            Return True, if this `Variable` is a constant (lower==upper),\
            False otherwise.

        """
        return self.up_bound == self.lo_bounds

    def __repr__(self):
        return (
            super(IntVar, self).__repr__()
            + f"\
        \t- Lower bound: {self.low_bound}\n\
        \t- Upper bound: {self.up_bound}\n"
        )


# Real
class FloatVar(Variable):
    """IntVar

    `IntVar` is a `Variable` discribing an Float variable.

    Parameters
    ----------
    label : str
        Name of the variable.
    lower : {int,float}
        Lower bound of the variable
    upper : {int,float}
        Upper bound of the variable

    Attributes
    ----------
    up_bound : {int,float}
        Lower bound of the variable
    low_bound : {int,float}
        Upper bound of the variable

    Examples
    --------
    >>> from zellij.core.variables import FloatVar
    >>> a = FloatVar("test", 0, 5.0)
    >>> print(a)
    FloatVar:
                - Label: test
                - Lower bound: 0
                - Upper bound: 5.0
    >>> a.random()
    2.2011985711663056

    """

    def __init__(self, label, lower, upper, sampler=np.random.uniform):
        super(FloatVar, self).__init__(label)

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
        """random(size=None)

        Parameters
        ----------
        size : int, default=None
            Number of draws.

        Returns
        -------
        out: float or list[float]
            Return a float if `size`=1, a list[float] else.

        """
        return self.sampler(self.low_bound, self.up_bound, size)

    def isconstant(self):
        """isconstant()

        Returns
        -------
        out: boolean
            Return True, if this `Variable` is a constant (lower==upper),\
            False otherwise.

        """
        return self.up_bound == self.lo_bounds

    def __repr__(self):
        return (
            super(FloatVar, self).__repr__()
            + f"\
        \t- Lower bound: {self.low_bound}\n\
        \t- Upper bound: {self.up_bound}\n"
        )


# Categorical
class CatVar(Variable):
    """CatVar(Variable)

    `CatVar` is a `Variable` discribing what a categorical variable is.

    Parameters
    ----------
    label : str
        Name of the variable.
    features : list
        List of all choices.
    weights : list[float]
        Wieghts associated to each elements of `features`. The sum of all
        positive elements of this list, must be equal to 1.

    Attributes
    ----------
    features
    weights

    Examples
    --------
    >>> from zellij.core.variables import CatVar, IntVar
    >>> a = CatVar("test", ['a', 1, 2.56, IntVar("int", 100 , 200)])
    >>> print(a)
    CatVar:
                - Label: test
                - Features: ['a', 1, 2.56,
    IntVar:
                - Label: int
                - Lower bound: 100
                - Upper bound: 200
    ]
    >>> a.random(10)
    ['a', 180, 2.56, 'a', 'a', 2.56, 185, 2.56, 105, 1]

    """

    def __init__(self, label, features, weights=None):
        super(CatVar, self).__init__(label)

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
        """random(size=1)

        Parameters
        ----------
        size : int, default=1
            Number of draws.

        Returns
        -------
        out: float or list[float]
            Return a feature if `size`=1, a list[features] else.
            If the feature is a `Variable` is the `random()` method from this
            `Variable`

        """

        if size == 1:
            res = random.choices(self.features, weights=self.weights, k=size)[0]
            if isinstance(res, Variable):
                res = res.random()
        else:
            res = random.choices(self.features, weights=self.weights, k=size)

            for i, v in enumerate(res):
                if isinstance(v, Variable):
                    res[i] = v.random()

        return res

    def isconstant(self):
        """isconstant()

        Returns
        -------
        out: boolean
            Return True, if this `Variable` is a constant (len(feature)==1),\
            False otherwise.

        """

        return len(self.features) == 1

    def __repr__(self):
        return (
            super(CatVar, self).__repr__()
            + f"\
        \t- Features: {self.features}\n"
        )


# Array of variables
class ArrayVar(Variable):
    """ArrayVar(Variable)

    `ArrayVar` is a `Variable` describing a list of `Variable`. This class is
    iterable.

    Parameters
    ----------
    label : str
        Name of the variable.
    *args : list[Variable]
        Elements of the `ArrayVar`. All elements must be of type `Variable`

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, IntVar, FloatVar, CatVar
    >>> a = ArrayVar("test",
    ...                     IntVar("int_1", 0,8),
    ...                     IntVar("int_2", 4,45),
    ...                     FloatVar("float_1", 2,12),
    ...                     CatVar("cat_1", ["Hello", 87, 2.56]))
    >>> print(a)
    ArrayVar:
                - Label: test
                - Length: 4
    =====
    [

    IntVar:
                - Label: int_1
                - Lower bound: 0
                - Upper bound: 8

    IntVar:
                - Label: int_2
                - Lower bound: 4
                - Upper bound: 45

    FloatVar:
                - Label: float_1
                - Lower bound: 2
                - Upper bound: 12

    CatVar:
                - Label: cat_1
                - Features: ['Hello', 87, 2.56]

    ]
    =====
    >>> a.random()
    [5, 15, 8.483221226216427, 'Hello']
    """

    def __init__(self, label, *args):
        super(ArrayVar, self).__init__(label)

        assert all(isinstance(v, Variable) for v in args), logger.error(
            f"All elements must inherit from `Variable`, got {args}"
        )
        self.values = args

    def random(self, size=1):
        """random(size=1)

        Parameters
        ----------
        size : int, default=None
            Number of draws.

        Returns
        -------
        out: float or list[float]
            Return a list composed of the values returned by each `Variable` of
            `ArrayVar`. If size>1, return a list of list

        """

        if size == 1:
            return [v.random() for v in self.values]
        else:
            res = []
            for _ in range(size):
                res.append([v.random() for v in self.values])

            return res

    def isconstant(self):
        """isconstant()

        Returns
        -------
        out: boolean
            Return True, if this `Variable` is a constant (all elements are
            constants), False otherwise.

        """
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

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        values_reprs = ""
        for v in self.values:
            values_reprs += v.__repr__()

        return (
            super(ArrayVar, self).__repr__()
            + f"\
        \t- Length: {len(self)}\n=====\n[\n"
            + values_reprs
            + "\n]\n=====\n"
        )


# Block of variable, fixed size
class Block(Variable):
    """Block(Variable)

    A `Block` is a `Variable` which will repeat multiple times a `Variable`.

    Parameters
    ----------
    label : str
        Name of the variable.
    value : Variable
        `Variable` that will be repeated
    repeat : int
        Number of repeats.

    Examples
    --------
    >>> from zellij.core.variables import Block, ArrayVar, FloatVar, IntVar
    >>> content = ArrayVar("test",
    ...                     IntVar("int_1", 0,8),
    ...                     IntVar("int_2", 4,45),
    ...                     FloatVar("float_1", 2,12))
    >>> a = Block("size 3 Block", content, 3)
    >>> print(a)
    Block:
                - Label: size 3 Block
         Block of:
    ArrayVar:
                - Label: test
                - Length: 3
    =====
    [

    IntVar:
                - Label: int_1
                - Lower bound: 0
                - Upper bound: 8

    IntVar:
                - Label: int_2
                - Lower bound: 4
                - Upper bound: 45

    FloatVar:
                - Label: float_1
                - Lower bound: 2
                - Upper bound: 12

    ]
    =====
    >>> a.random(3)
    [[[7, 22, 6.843164591359903],
        [5, 18, 10.608957810018786],
        [4, 21, 10.999649079045858]],
    [[5, 9, 9.773288692746476],
        [1, 12, 6.1909724243671445],
        [4, 12, 9.404313234593669]],
    [[4, 10, 2.72648188721585],
        [1, 44, 5.319257221471118],
        [4, 24, 9.153357213126071]]]

    """

    def __init__(self, label, value, repeat):
        super(Block, self).__init__(label)

        assert isinstance(value, Variable), logger.error(
            f"Value must inherit from `Variable`, got {args}"
        )
        self.value = value

        assert isinstance(repeat, int) and repeat > 0, logger.error(
            f"`repeat` must be a strictly positive int, got {repeat}."
        )
        self.repeat = repeat

    def random(self, size=1):
        """random(size=1)

        Parameters
        ----------
        size : int, default=None
            Number of draws.

        Returns
        -------
        out: float or list[float]
            Return a list composed of the results from the `Variable` `random()`
            method, repeated `repeat` times. If size > 1, return a list of list.

        """

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
        """isconstant()

        Returns
        -------
        out: boolean
            Return True, if this `Variable` is a constant (the repeated
            `Variable` is constant), False otherwise.

        """

        return self.value.isconstant()

    def __repr__(self):
        values_reprs = ""
        for v in self.value:
            values_reprs += v.__repr__()

        return (
            super(Block, self).__repr__()
            + f"\t Block of:{self.value.__repr__()}\n"
        )


# Block of variables, with random size.
class DynamicBlock(Block):
    """DynamicBlock(Block)

    A `DynamicBlock` is a `Block` with a random number of repeats.

    Parameters
    ----------
    label : str
        Name of the variable.
    value : Variable
        `Variable` that will be repeated
    repeat : int
        Maximum number of repeats.

    Examples
    --------
    >>> from zellij.core.variables import DynamicBlock, ArrayVar, FloatVar, IntVar
    >>> content = ArrayVar("test",
    ...                     IntVar("int_1", 0,8),
    ...                     IntVar("int_2", 4,45),
    ...                     FloatVar("float_1", 2,12))
    >>> a = DynamicBlock("max size 10 Block", content, 10)
    >>> print(a)
    DynamicBlock:
                - Label: max size 10 Block
         Block of:
    ArrayVar:
                - Label: test
                - Length: 3
    =====
    [

    IntVar:
                - Label: int_1
                - Lower bound: 0
                - Upper bound: 8

    IntVar:
                - Label: int_2
                - Lower bound: 4
                - Upper bound: 45

    FloatVar:
                - Label: float_1
                - Lower bound: 2
                - Upper bound: 12

    ]
    =====
    >>> a.random()
    [[[3, 12, 10.662362255103403],
          [7, 9, 5.496860842510198],
          [3, 37, 7.25449459082227],
          [4, 28, 4.912883181322568]],
    [[3, 23, 5.150228671772998]],
    [[6, 30, 6.1181372194738515]]]

    """

    def __init__(self, label, value, repeat):
        super(DynamicBlock, self).__init__(label, value, repeat)

    def random(self, size=1):
        """random(size=1)

        Parameters
        ----------
        size : int, default=None
            Number of draws.

        Returns
        -------
        out: float or list[float]
            Return a list composed of the results from the `Variable` `random()`
            method, repeated `repeat` times. If size > 1, return a list of list.

        """
        res = []

        if size > 1:
            for _ in range(size):
                block = []
                n_repeat = np.random.randint(1, self.repeat)
                for _ in range(n_repeat):
                    block.append([v.random() for v in self.value])
                res.append(block)
        else:
            n_repeat = np.random.randint(1, self.repeat)
            for _ in range(n_repeat):
                res.append([v.random() for v in self.value])

        return res

    def isconstant(self):
        """isconstant()

        Returns
        -------
        out: False
            Return False, a dynamic block cannot be constant. (It is a binary)

        """
        return False
