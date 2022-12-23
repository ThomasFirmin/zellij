# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-12T11:18:26+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:36:55+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.addons import VarAddon
from abc import ABC, abstractmethod
import math
import numpy as np
import random
import copy

import logging

logger = logging.getLogger("zellij.variables")
logger.setLevel(logging.INFO)


@abstractmethod
class Variable(ABC):
    """Variable

    :ref:`var` is an Abstract class defining what a variable is in a :ref:`sp`.

    Parameters
    ----------
    label : str
        Name of the variable.
    kwargs : dict
        Kwargs will be the different addons you want to add to a :ref:`var`.
        Known addons are:
        * to_discrete : VarConverter
        * to_continuous : VarConverter
        * neighbor : VarNeighborhood

    Attributes
    ----------
    label
    """

    def __init__(self, label, **kwargs):
        assert isinstance(
            label, str
        ), f"""
        Label must be a string, got {label}
        """

        self.label = label
        self.kwargs = kwargs

        self._add_addons(**kwargs)

    @abstractmethod
    def random(self, size=None):
        pass

    @abstractmethod
    def isconstant(self):
        pass

    @abstractmethod
    def subset(self):
        pass

    # @abstractmethod
    # def header(self):
    #     pass
    #
    # @abstractmethod
    # def save(self, x):
    #     pass

    def _add_addons(self, **kwargs):
        for k in kwargs:

            assert isinstance(
                kwargs[k], VarAddon
            ), f"""
            Kwargs must be of type `VarAddon`, got {k}:{kwargs[k]}
            """
            if kwargs[k]:
                setattr(self, k, copy.copy(kwargs[k]))
                addon = getattr(self, k)
                addon.target = self
            else:
                setattr(self, k, kwargs[k])
                addon = getattr(self, k)
                addon.target = self

    def __repr__(self):
        return f"{self.__class__.__name__}({self.label}, "


# Discrete
class IntVar(Variable):
    """IntVar

    `IntVar` is a :ref:`var` discribing an Integer variable. T
    he :code:`lower` and :code:`upper` bounds are included.

    Parameters
    ----------
    label : str
        Name of the variable.
    lower : int
        Lower bound of the variable
    upper : int
        Upper bound of the variable
    sampler : Callable, default=np.random.randint
        Function that takes lower bound, upper bound and a size as parameters.

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
    IntVar(test, [0;5])
    >>> a.random()
    1

    """

    def __init__(
        self, label, lower, upper, sampler=np.random.randint, **kwargs
    ):
        super(IntVar, self).__init__(label, **kwargs)

        assert isinstance(
            upper, (int, np.integer)
        ), f"""
        Upper bound must be an int, got {upper}
        """

        assert isinstance(
            lower, (int, np.integer)
        ), f"""
        Lower bound must be an int, got {lower}
        """

        assert (
            lower < upper
        ), f"""Lower bound must be
        strictly inferior to upper bound,  got {lower}<{upper}"""

        self.low_bound = lower
        self.up_bound = upper + 1
        self.sampler = sampler

    def random(self, size=None):
        """random(size=None)

        Parameters
        ----------
        size : int, default=None
            Number of draws.

        Returns
        -------
        out: int or list[int]
            Return an int if :code:`size`=1, a :code:`list[int]` else.

        """
        return self.sampler(self.low_bound, self.up_bound, size, dtype=int)

    def isconstant(self):
        """isconstant()

        Returns
        -------
        out: boolean
            Return True, if this :ref:`var` is a constant
            (:code:`lower`==:code:`upper`),\
            False otherwise.

        """
        return self.up_bound == self.lo_bounds

    def subset(self, lower, upper):
        assert isinstance(
            upper, (int, np.integer)
        ), f"""Upper bound must be an int, got {upper}"""
        assert isinstance(
            lower, (int, np.integer)
        ), f"""Upper bound must be an int, got {lower}"""
        assert (
            lower >= self.low_bound
        ), f"""
        Subset lower bound must be higher than the initial lower bound,
         got {lower}>{self.low_bound}
        """

        assert (
            upper <= self.up_bound
        ), f"""
        Subset upper bound must be lower than the initial upper bound,
         got {lower}<{upper}
        """

        if upper == lower:
            return Constant(self.label, lower)
        else:
            return IntVar(self.label, lower, upper)

    def __len__(self):
        return 1

    def __repr__(self):
        return (
            super(IntVar, self).__repr__()
            + f"[{self.low_bound};{self.up_bound}])"
        )


# Real
class FloatVar(Variable):
    """FloatVar

    `FloatVar` is a :ref:`var` discribing a Float variable.

    Parameters
    ----------
    label : str
        Name of the variable.
    lower : {int,float}
        Lower bound of the variable
    upper : {int,float}
        Upper bound of the variable
    sampler : Callable, default=np.random.uniform
        Function that takes lower bound, upper bound and a size as parameters.

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
    FloatVar(test, [0;5.0])
    >>> a.random()
    2.2011985711663056

    """

    def __init__(
        self,
        label,
        lower,
        upper,
        sampler=np.random.uniform,
        tolerance=1e-14,
        **kwargs,
    ):
        super(FloatVar, self).__init__(label, **kwargs)

        assert isinstance(
            upper, (float, int, np.integer, np.floating)
        ), f"""Upper bound must be an int or a float, got {upper}"""

        assert isinstance(
            lower, (float, int, np.integer, np.floating)
        ), f"""Lower bound must be an int or a float, got {lower}"""

        assert (
            lower < upper
        ), f"""Lower bound must be
         strictly inferior to upper bound, got {lower}<{upper}"""

        assert tolerance >= 0, f"""Tolerance must be > 0, got{tolerance}"""

        self.up_bound = upper
        self.low_bound = lower
        self.sampler = sampler
        self.tolerance = tolerance

    def random(self, size=None):
        """random(size=None)

        Parameters
        ----------
        size : int, default=None
            Number of draws.

        Returns
        -------
        out: float or list[float]
            Return a float if :code:`size`=1, a :code:`list[float]` else.

        """
        return self.sampler(self.low_bound, self.up_bound, size)

    def isconstant(self):
        """isconstant()

        Returns
        -------
        out: boolean
            Return True, if this :ref:`var` is a constant
            (:code:`lower`==:code:`upper`),\
            False otherwise.

        """
        return self.up_bound == self.lo_bounds

    def subset(self, lower, upper):
        assert isinstance(
            upper, (float, int, np.integer, np.floating)
        ), f"""
        Upper bound must be an int, got {upper}
        """

        assert isinstance(lower, int) or isinstance(
            lower, (float, int, np.integer, np.floating)
        ), f"""
        Upper bound must be an int, got {lower}
        """

        assert (
            lower - self.low_bound >= -self.tolerance
            and lower - self.up_bound <= self.tolerance
        ), f"""
        Subset lower bound must be higher than the initial lower bound,
        got {lower}>={self.low_bound}
        """

        assert (
            upper - self.up_bound <= self.tolerance
            and upper - self.low_bound >= -self.tolerance
        ), f"""
        Subset upper bound must be lower than the initial upper bound,
        got {upper}<={self.up_bound}
        """

        if math.isclose(upper, lower, abs_tol=self.tolerance):
            return Constant(self.label, float(lower))
        else:
            return FloatVar(
                self.label,
                lower,
                upper,
                sampler=self.sampler,
                tolerance=self.tolerance,
                **self.kwargs,
            )

    def __len__(self):
        return 1

    def __repr__(self):
        return (
            super(FloatVar, self).__repr__()
            + f"[{self.low_bound};{self.up_bound}])"
        )


# Categorical
class CatVar(Variable):
    """CatVar(Variable)

    `CatVar` is a :ref:`var` discribing what a categorical variable is.

    Parameters
    ----------
    label : str
        Name of the variable.
    features : list
        List of all choices.
    weights : list[float]
        Weights associated to each elements of :code:`features`. The sum of all
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
    CatVar(test, ['a', 1, 2.56, IntVar(int, [100;200])])
    >>> a.random(10)
    ['a', 180, 2.56, 'a', 'a', 2.56, 185, 2.56, 105, 1]

    """

    def __init__(self, label, features, weights=None, **kwargs):
        super(CatVar, self).__init__(label, **kwargs)

        assert isinstance(
            features, list
        ), f"""
        Features must be a list with a length > 0, got{features}
        """

        assert (
            len(features) > 1
        ), f"""
        Features must be a list with a length > 1,
        got length= {len(features)}
        """

        self.features = features

        assert (
            isinstance(weights, (list, np.ndarray)) or weights == None
        ), f"""`weights` must be a list or equal to None, got {weights}"""

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
            Return a feature if :code:`size`=1, a :code:`list[features]` else.
            Features can be :ref:`var`. When seleted, it will return
            a random point from this :ref:`var`.

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
            Return True, if this :ref:`var` is a constant
            (:code:`len(feature)==1`),\
            False otherwise.

        """

        return len(self.features) == 1

    def subset(self, lower, upper):
        assert (
            upper in self.features
        ), f"""
        Upper bound is not in features of CatVar, got {upper}"""

        assert (
            lower in self.features
        ), f"""
        Lower bound is not in features of CatVar, got {lower}"""

        if upper == lower:
            return Constant(self.label, lower)
        else:

            lo_idx = self.features.index(lower)
            up_idx = self.features.index(upper)

            if lo_idx > up_idx:
                return CatVar(
                    self.label,
                    self.features[lo_idx:] + self.features[: up_idx + 1],
                )
            else:
                return CatVar(self.label, self.features[lo_idx : up_idx + 1])

    def __len__(self):
        return 1

    def __repr__(self):
        return super(CatVar, self).__repr__() + f"{self.features})"


# Array of variables
class ArrayVar(Variable):
    """ArrayVar(Variable)

    :code:`ArrayVar` is a :ref:`var` describing a list of :ref:`var`. This class is
    iterable.

    Parameters
    ----------
    label : str
        Name of the variable.
    *args : list[Variable]
        Elements of the :code:`ArrayVar`. All elements must be of type :ref:`var`

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, IntVar, FloatVar, CatVar
    >>> a = ArrayVar(IntVar("int_1", 0,8),
    ...              IntVar("int_2", 4,45),
    ...              FloatVar("float_1", 2,12),
    ...              CatVar("cat_1", ["Hello", 87, 2.56]))
    >>> print(a)
    ArrayVar(, [IntVar(int_1, [0;8]),
                IntVar(int_2, [4;45]),
                FloatVar(float_1, [2;12]),
                CatVar(cat_1, ['Hello', 87, 2.56])])
    >>> a.random()
    [5, 15, 8.483221226216427, 'Hello']
    """

    def __init__(self, *args, label="", **kwargs):

        if args and len(args) > 1 and args[0]:
            assert all(
                isinstance(v, Variable) for v in args
            ), f"""
            All elements must inherit from :ref:`var`,
            got {args}
            """

            self.values = list(args)
            for idx, v in enumerate(self.values):
                setattr(v, "_idx", idx)
        else:
            self.values = []

        super(ArrayVar, self).__init__(label, **kwargs)

    def random(self, size=1):
        """random(size=1)

        Parameters
        ----------
        size : int, default=None
            Number of draws.

        Returns
        -------
        out: float or list[float]
            Return a list composed of the values returned by each :ref:`var` of
            :code:`ArrayVar`. If :code:`size`>1, return a list of list

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
            Return True, if this :ref:`var` is a constant (all elements are
            constants), False otherwise.

        """
        return all(v.isconstant for v in self.values)

    def subset(self, lower, upper):
        assert isinstance(lower, (list, np.ndarray)) and (
            len(lower) == len(self)
        ), f"""
            Lower bound must be a list containing lower bound of each
            :ref:`var` composing :code:`ArrayVar`, got {lower}
            """

        assert isinstance(upper, (list, np.ndarray)) and (
            len(upper) == len(self)
        ), f"""
        Upper bound must be a list containing lower bound of each
        :ref:`var` composing :code:`ArrayVar`, got {upper}
        """

        new_values = []
        for v, l, u in zip(self.values, lower, upper):
            new_values.append(v.subset(l, u))

        return ArrayVar(*new_values, label=self.label, **self.kwargs)

    def index(self, value):
        """index(value)

        Return the index inside the :code::code:`ArrayVar` of a given :code:`value`.

        Parameters
        ----------
        value : Variable
            Targeted Variable in the ArrayVar

        Returns
        -------
        int
            Index of :code:`value`.

        """
        return value._idx

    def append(self, v):
        """append(v)

        Append a :ref:`Variables` to the :code::code:`ArrayVar`.

        Parameters
        ----------
        v : Variable
            Variable to be added to the :code:`ArrayVar`

        """
        if isinstance(v, Variable):
            setattr(v, "_idx", len(self.values))
            self.values.append(v)
        else:
            raise ValueError(
                f"""
            Cannot append a {type(v)} to ArrayVar.
            Tried to append {v} to {self}.
            """
            )

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):

        if self.index >= len(self.values):
            raise StopIteration

        res = self.values[self.index]
        self.index += 1
        return res

    def __getitem__(self, item):
        return self.values[item]

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        values_reprs = ""
        for v in self.values:
            values_reprs += v.__repr__() + ","

        return super(ArrayVar, self).__repr__() + f"[{values_reprs[:-1]}])"


# Block of variable, fixed size
class Block(Variable):
    """Block(Variable)

    A `Block` is a :ref:`var` which will repeat multiple times a :ref:`var`.

    Parameters
    ----------
    label : str
        Name of the variable.
    value : Variable
        :ref:`var` that will be repeated
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
    Block(size 3 Block, [IntVar(int_1, [0;8]),
                         IntVar(int_2, [4;45]),
                         FloatVar(float_1, [2;12]),])
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

    def __init__(self, label, value, repeat, **kwargs):
        super(Block, self).__init__(label, **kwargs)

        assert isinstance(
            value, Variable
        ), f"""
        Value must inherit from :ref:`var`, got {args}
        """

        self.value = value

        assert (
            isinstance(repeat, int) and repeat > 0
        ), f"""
        `repeat` must be a strictly positive int, got {repeat}.
        """

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
            Return a list composed of the results from the :ref:`var` `random()`
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
            Return True, if this :ref:`var` is a constant (the repeated
            :ref:`var` is constant), False otherwise.

        """

        return self.value.isconstant()

    def subset(self, lower, upper):

        new_values = v.subset(l, u)

        return Block(self.label, new_values)

    def __repr__(self):
        values_reprs = ""
        for v in self.value:
            values_reprs += v.__repr__() + ","

        return super(Block, self).__repr__() + f"[{values_reprs}])"


# Block of variables, with random size.
class DynamicBlock(Block):
    """DynamicBlock(Block)

    A `DynamicBlock` is a `Block` with a random number of repeats.

    Parameters
    ----------
    label : str
        Name of the variable.
    value : Variable
        :ref:`var` that will be repeated
    repeat : int
        Maximum number of repeats.

    Examples
    --------
    >>> from zellij.core.variables import DynamicBlock, ArrayVar, FloatVar, IntVar
    >>> content = ArrayVar(IntVar("int_1", 0,8),
    ...                    IntVar("int_2", 4,45),
    ...                    FloatVar("float_1", 2,12))
    >>> a = DynamicBlock("max size 10 Block", content, 10)
    >>> print(a)
    DynamicBlock(max size 10 Block, [IntVar(int_1, [0;8]),
                                     IntVar(int_2, [4;45]),
                                     FloatVar(float_1, [2;12]),])
    >>> a.random()
    [[[3, 12, 10.662362255103403],
          [7, 9, 5.496860842510198],
          [3, 37, 7.25449459082227],
          [4, 28, 4.912883181322568]],
    [[3, 23, 5.150228671772998]],
    [[6, 30, 6.1181372194738515]]]

    """

    def __init__(self, label, value, repeat, **kwargs):
        super(DynamicBlock, self).__init__(label, value, repeat, **kwargs)

    def random(self, size=1):
        """random(size=1)

        Parameters
        ----------
        size : int, default=None
            Number of draws.

        Returns
        -------
        out: float or list[float]
            Return a list composed of the results from the :ref:`var` `random()`
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


# Constant
class Constant(Variable):
    """Constant

    :code:`Constant` is a :ref:`var` discribing a constant of any type.

    Parameters
    ----------
    label : str
        Name of the variable.
    value : object
        Constant value

    Attributes
    ----------
    label : str
        Name of the variable.
    value : object
        Constant value

    Examples
    --------
    >>> from zellij.core.variables import Constant
    >>> a = Constant("test", 5)
    >>> print(a)
    Constant(test, 5)
    >>> a.random()
    5

    """

    def __init__(self, label, value, **kwargs):
        super(Constant, self).__init__(label, **kwargs)
        assert not isinstance(
            value, Variable
        ), f"Element must not be of Variable type, got {value}"
        self.value = value

    def random(self, size=1):
        """random(size=None)

        Parameters
        ----------
        size : int, default=None
            Number of draws.

        Returns
        -------
        out: int or list[int]
            Return an int if :code:`size`=1, a :code:`list[self.value]` else.

        """
        if size > 1:
            return [self.value] * size
        else:
            return self.value

    def isconstant(self):
        """isconstant()

        Returns
        -------
        out: boolean
            Return True

        """
        return True

    def subset(self, l, u):
        return self

    def __repr__(self):
        return super(Constant, self).__repr__() + f"{self.value})"
