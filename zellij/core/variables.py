# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Callable, Union, List, TYPE_CHECKING
from dataclasses import dataclass

from zellij.core.addons import (
    VarConverter,
    VarNeighborhood,
    IntConverter,
    FloatConverter,
    CatConverter,
    ArrayConverter,
    PermutationConverter,
    IntNeighborhood,
    FloatNeighborhood,
    CatNeighborhood,
    ArrayNeighborhood,
    PermutationNeighborhood,
)

from zellij.core.errors import InitializationError

import numpy as np
import random

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
    converter : VarConverter, optional
        :code:`VarConverter` use to convert a given value from a :ref:`var` to another.
    neighborhood : VarNeighborhood, optional
        :code:`VarNeighborhood` used to define the neighborhood of a given value for a :ref:`var`.
        Used in :ref:`meta` using neighborhood. Such as :code:`SimulatedAnnealing`.

    Attributes
    ----------
    label
    converter
    neighborhood
    """

    def __init__(
        self,
        label: str,
        converter: Optional[VarConverter] = None,
        neighborhood: Optional[VarNeighborhood] = None,
    ):
        self.label = label
        self.converter = converter
        self.neighborhood = neighborhood

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str):
        if isinstance(value, str):
            self._label = value
        else:
            raise InitializationError(f"Label must be a string. Got {type(value)}.")

    @property
    def converter(self) -> Optional[VarConverter]:
        return self._converter

    @converter.setter
    @abstractmethod
    def converter(self, value: Optional[VarConverter]):
        pass

    @property
    def neighborhood(self) -> Optional[VarNeighborhood]:
        return self._neighborhood

    @neighborhood.setter
    @abstractmethod
    def neighborhood(self, value: Optional[VarNeighborhood]):
        pass

    def _converter_setter(self, value: Optional[VarConverter], atype: type):
        if isinstance(value, atype) or (value is None):
            self._converter = value
            if self._converter:
                self._converter.target = self
        else:
            raise InitializationError(
                f"In Variable, converter must be of type {atype} or NoneType. Got {type(value)}"
            )

    def _neighborhood_setter(self, value: Optional[VarNeighborhood], atype: type):
        if isinstance(value, atype) or (value is None):
            self._neighborhood = value
            if self._neighborhood:
                self._neighborhood.target = self
        else:
            raise InitializationError(
                f"In Variable, neighborhood must be of type {atype} or NoneType. Got {type(value)}"
            )

    @abstractmethod
    def random(self, size=None):
        pass

    @abstractmethod
    def __len__(self):
        pass

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
    sampler : Callable[low, high, size], default=np.random.randint
        Function that takes lower bound, upper bound and a size as parameters.
    converter : IntConverter, optional
        :code:`IntConverter` use to convert a given value from a :ref:`var` to another.
    neighborhood : IntNeighborhood, optional
        :code:`IntNeighborhood` used to define the neighborhood of a given value for a :ref:`var`.
        Used in :ref:`meta` using neighborhood. Such as :code:`SimulatedAnnealing`.

    Attributes
    ----------
    up_bound : int
        Lower bound of the variable
    low_bound : int
        Upper bound of the variable

    Examples
    --------
    >>> from zellij.core import IntVar
    >>> a = IntVar("i1",0,5)
    >>> print(a)
    IntVar(test, [0;5])
    >>> print(a.label, a.lower, a.upper)
    i1 0 5
    >>> a.random()
    1
    >>> a.random(5)
    array([1, 5, 4, 2, 4])
    """

    def __init__(
        self,
        label,
        lower: int,
        upper: int,
        sampler: Callable[
            [int, int, Optional[int]], Union[int, List[int], np.ndarray]
        ] = np.random.randint,
        converter: Optional[IntConverter] = None,
        neighborhood: Optional[IntNeighborhood] = None,
    ):
        self._lower = float("-inf")
        self._upper = float("inf")

        self.upper = upper
        self.lower = lower

        self.sampler = sampler

        super(IntVar, self).__init__(
            label, converter=converter, neighborhood=neighborhood
        )

    @Variable.converter.setter
    def converter(self, value: Optional[IntConverter]):
        self._converter_setter(value, IntConverter)

    @Variable.neighborhood.setter
    def neighborhood(self, value: Optional[IntNeighborhood]):
        self._neighborhood_setter(value, IntNeighborhood)

    @property
    def lower(self) -> int:
        return self._lower  # type: ignore

    @lower.setter
    def lower(self, value: int):
        if isinstance(value, int) and value < self._upper:
            self._lower = value
        else:
            raise InitializationError(f"Lower must be an int and < upper. Got {value}")

    @property
    def upper(self) -> int:
        return self._upper  # type: ignore

    @upper.setter
    def upper(self, value: int):
        if isinstance(value, int) and value > self._lower:
            self._upper = value
        else:
            raise InitializationError(f"Upper must be an int and > lower. Got {value}")

    def random(self, size: Optional[int] = None):
        """random(size=None)

        Parameters
        ----------
        size : int, optional
            Number of draws. If None, returns a single int.

        Returns
        -------
        out: int or list[int]
            Return an int if :code:`size`=1, a :code:`list[int]` else.

        """
        return self.sampler(self.lower, self.upper, size)

    def __len__(self):
        return 1

    def __repr__(self):
        return super(IntVar, self).__repr__() + f"[{self.lower};{self.upper}])"


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
    sampler : Callable[low, high, size], default=np.random.uniform
        Function that takes lower bound, upper bound and a size as parameters.
    converter : FloatConverter, optional
        :code:`FloatConverter` use to convert a given value from a :ref:`var` to another.
    neighborhood : FloatNeighborhood, optional
        :code:`FloatNeighborhood` used to define the neighborhood of a given value for a :ref:`var`.
        Used in :ref:`meta` using neighborhood. Such as :code:`SimulatedAnnealing`.

    Attributes
    ----------
    up_bound : {int,float}
        Lower bound of the variable
    low_bound : {int,float}
        Upper bound of the variable

    Examples
    --------
    >>> from zellij.core import FloatVar
    >>> a = FloatVar("f1",0,5)
    >>> print(a)
    FloatVar(f1, [0.0;5.0])
    >>> print(a.label, a.lower, a.upper)
    f1 0.0 5.0
    >>> a.random()
    1.4650916221213444
    >>> a.random(5)
    array([4.37946741 4.12848337 3.50995715 1.16979835 3.46117219])
    """

    def __init__(
        self,
        label,
        lower: float,
        upper: float,
        sampler: Callable[
            [float, float, Optional[int]], Union[float, List[float], np.ndarray]
        ] = np.random.uniform,
        converter: Optional[FloatConverter] = None,
        neighborhood: Optional[FloatNeighborhood] = None,
    ):
        self._lower = float("-inf")
        self._upper = float("inf")

        self.upper = upper
        self.lower = lower

        self.sampler = sampler

        super(FloatVar, self).__init__(
            label, converter=converter, neighborhood=neighborhood
        )

    @Variable.converter.setter
    def converter(self, value: Optional[FloatConverter]):
        self._converter_setter(value, FloatConverter)

    @Variable.neighborhood.setter
    def neighborhood(self, value: Optional[FloatNeighborhood]):
        self._neighborhood_setter(value, FloatNeighborhood)

    @property
    def lower(self) -> float:
        return self._lower

    @lower.setter
    def lower(self, value: float | int):
        if isinstance(value, (float, int)) and value < self._upper:
            self._lower = float(value)
        else:
            raise InitializationError(f"Lower must be a float and < upper. Got {value}")

    @property
    def upper(self) -> float:
        return self._upper

    @upper.setter
    def upper(self, value: float | int):
        if isinstance(value, (float, int)) and value > self._lower:
            self._upper = float(value)
        else:
            raise InitializationError(f"Upper must be an int and > lower. Got {value}")

    def random(self, size: Optional[int] = None):
        """random

        Parameters
        ----------
        size : int, optional
            Number of draws. If None, returns a single float.

        Returns
        -------
        out: float or list[float]
            Return a float if :code:`size`=None, a :code:`list[float]` else.

        """
        return self.sampler(self.lower, self.upper, size)

    def __len__(self):
        return 1

    def __repr__(self):
        return super(FloatVar, self).__repr__() + f"[{self.lower};{self.upper}])"


# Categorical
class CatVar(Variable):
    """CatVar(Variable)

    `CatVar` is a :ref:`var` discribing what a categorical variable is.

    Parameters
    ----------
    label : str
        Name of the variable.
    features : list
        List of all possible features.
    weights : ndarray[float], optional
        Weights associated to each elements of :code:`features`. The sum of all
        positive elements of this list, must be equal to 1.
    converter : CatConverter, optional
        :code:`CatConverter` use to convert a given value from a :ref:`var` to another.
    neighborhood : CatNeighborhood, optional
        :code:`CatNeighborhood` used to define the neighborhood of a given value for a :ref:`var`.
        Used in :ref:`meta` using neighborhood. Such as :code:`SimulatedAnnealing`.

    Attributes
    ----------
    features
    weights

    Examples
    --------
    >>> from zellij.core import CatVar
    >>> a = CatVar("c1",["a","b","c"])
    >>> print(a)
    CatVar(c1, ['a', 'b', 'c'])
    >>> print(a.label, a.features)
    c1 ['a', 'b', 'c']
    >>> a.random()
    'a'
    >>> a.random(5)
    ['c', 'c', 'c', 'a', 'b']
    """

    def __init__(
        self,
        label: str,
        features: list,
        weights: Optional[Union[List[float], np.ndarray]] = None,
        converter: Optional[CatConverter] = None,
        neighborhood: Optional[CatNeighborhood] = None,
    ):
        self.features = features
        self.weights = weights

        super(CatVar, self).__init__(
            label, converter=converter, neighborhood=neighborhood
        )

    @Variable.converter.setter
    def converter(self, value: Optional[CatConverter]):
        self._converter_setter(value, CatConverter)

    @Variable.neighborhood.setter
    def neighborhood(self, value: Optional[CatNeighborhood]):
        self._neighborhood_setter(value, CatNeighborhood)

    @property
    def features(self) -> list:
        return self._features

    @features.setter
    def features(self, value: list):
        if isinstance(value, list) and len(value) > 1:
            self._features = value
        else:
            raise InitializationError(
                f"Features must be a list with a length > 0, got{value}"
            )

    @property
    def weights(self) -> List[float]:
        return self._weights

    @weights.setter
    def weights(self, value: Optional[Union[List[float], np.ndarray]]):
        if value is None:
            size = len(self.features)
            self.weights = [1 / size] * size
        elif isinstance(value, (list, np.ndarray)) and len(value) == len(self.features):
            self._weights = list(value)
        else:
            raise InitializationError(
                "weights must be a list of float, a numpy.ndarray, or None. Length of weights must be equal to length of features."
            )

    def random(self, size: Optional[int] = None):
        """random

        Parameters
        ----------
        size : int, optional
            Number of draws. If None, returns a single value.

        Returns
        -------
        out: object
            Return a random feature or list of random features.
        """

        if size:
            res = random.choices(self.features, weights=self.weights, k=size)
        else:
            res = random.choices(self.features, weights=self.weights, k=1)[0]

        return res

    def __len__(self):
        return 1

    def __repr__(self):
        return super(CatVar, self).__repr__() + f"{self.features})"


class IterableVar(Variable):
    @abstractmethod
    def random(self, size: Optional[int]) -> list:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


# Array of variables
class ArrayVar(IterableVar):
    """ArrayVar

    :code:`ArrayVar` is a :ref:`var` describing a list of :ref:`var`. This class is
    iterable.

    Parameters
    ----------
    args : list[Variable]
        Elements of the :code:`ArrayVar`. All elements must be of type :ref:`var`
    label : str
        Name of the variable.
    converter : ArrayConverter, optional
        :code:`ArrayConverter` use to convert a given value from a :ref:`var` to another.
    neighborhood : ArrayNeighborhood, optional
        :code:`ArrayNeighborhood` used to define the neighborhood of a given value for a :ref:`var`.
        Used in :ref:`meta` using neighborhood. Such as :code:`SimulatedAnnealing`.

    Examples
    --------
    >>> from zellij.core import ArrayVar, IntVar, FloatVar, CatVar
    >>> a = ArrayVar(IntVar("i1", 0,8),
    ...              IntVar("i2", 30,40),
    ...              FloatVar("f1", 10,20),
    ...              CatVar("c1", ["Hello", 87, 2.56]))
    >>> print(a)
    ArrayVar(, [IntVar(i1, [0;8]),IntVar(i2, [30;40]),FloatVar(f1, [10.0;20.0]),CatVar(c1, ['Hello', 87, 2.56])])
    >>> print(f"Label {a.label}, Length {len(a)}, Variables {a.variables}")
    Label , Length 4, Variables [IntVar(i1, [0;8]), IntVar(i2, [30;40]), FloatVar(f1, [10.0;20.0]), CatVar(c1, ['Hello', 87, 2.56])]
    >>> a.random()
    [6, 35, 12.920699881017159, 'Hello']
    >>> a.random(5)
    [[1, 31, 13.402872157435267, 'Hello'], [0, 38, 12.955169573740326, 'Hello'], [4, 30, 18.54926067947143, 'Hello'], [1, 32, 17.905372329473572, 'Hello'], [8, 34, 19.446512508103357, 'Hello']]
    >>> for v in a:
    ...     print(v)
    IntVar(i1, [0;8])
    IntVar(i2, [30;40])
    FloatVar(f1, [10.0;20.0])
    CatVar(c1, ['Hello', 87, 2.56])
    >>> new = IntVar("i3", 100,200)
    >>> a.append(new)
    >>> print(a)
    ArrayVar(, [IntVar(i1, [0;8]), IntVar(i2, [30;40]), FloatVar(f1, [10.0;20.0]), CatVar(c1, ['Hello', 87, 2.56]), IntVar(i3, [100;200])])
    """

    def __init__(
        self,
        *args: Variable,
        label: str = "",
        converter: Optional[ArrayConverter] = None,
        neighborhood: Optional[ArrayNeighborhood] = None,
    ):
        self.variables = list(args)
        self._index = 0

        super(ArrayVar, self).__init__(
            label, converter=converter, neighborhood=neighborhood
        )

    @Variable.converter.setter
    def converter(self, value: Optional[ArrayConverter]):
        self._converter_setter(value, ArrayConverter)

    @Variable.neighborhood.setter
    def neighborhood(self, value: Optional[ArrayNeighborhood]):
        self._neighborhood_setter(value, ArrayNeighborhood)

    @property
    def variables(self) -> List[Variable]:
        return self._variables

    @variables.setter
    def variables(self, value: List[Variable]):
        if isinstance(value, list) and len(value) > 1:
            if all(isinstance(v, Variable) for v in value):
                self._variables = list(value)
            else:
                raise InitializationError(
                    f"All elements must inherit from :ref:`var`,got {value}"
                )
        else:
            self._variables = []

    def random(self, size: Optional[int] = None) -> list:
        """random

        Parameters
        ----------
        size : int, optional
            Number of draws. If None, returns a single list.

        Returns
        -------
        out: list
            Return a single list if size is None, list of values from :code:`variables` else.
        """

        if size:
            res = []
            for _ in range(size):
                res.append([v.random() for v in self.variables])
            return res
        else:
            return [v.random() for v in self.variables]

    def index(self, var: Variable):
        """index(var)

        Return the index inside the :code:`ArrayVar` of a given :code:`var`.

        Parameters
        ----------
        var : Variable
            Targeted Variable in the ArrayVar

        Returns
        -------
        int
            Index of :code:`var`.

        """
        return self.variables.index(var)

    def append(self, v: Variable):
        """append(v)

        Append a :ref:`var` to the :code:`ArrayVar`.

        Parameters
        ----------
        v : Variable
            Variable to be added to the :code:`ArrayVar`

        """
        if isinstance(v, Variable):
            self.variables.append(v)
        else:
            raise ValueError(
                f"""
            Cannot append a {type(v)} to ArrayVar.
            Tried to append {v} to {self}.
            """
            )

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self) -> Variable:
        if self._index >= len(self.variables):
            raise StopIteration

        res = self.variables[self._index]
        self._index += 1
        return res

    def __getitem__(self, key):
        return self.variables[key]

    def __len__(self):
        return len(self.variables)

    def __repr__(self):
        vars_reprs = ""
        for v in self.variables:
            vars_reprs += v.__repr__() + ","

        return super(ArrayVar, self).__repr__() + f"[{vars_reprs[:-1]}])"


@dataclass
class _PermElem:
    label: str


# Array of variables
class PermutationVar(IterableVar):
    """PermutationVar

    :code:`ArrayVar` is a :ref:`var` describing a list permutation of unique integers. This class is
    iterable.

    Parameters
    ----------
    n : int
        Number of elements within the permutation.
    label : str
        Name of the variable.
    converter : PermutationConverter, optional
        :code:`PermutationConverter` use to convert a given value from a :ref:`var` to another.
    neighborhood : PermutationNeighborhood, optional
        :code:`PermutationNeighborhood` used to define the neighborhood of a given value for a :ref:`var`.
        Used in :ref:`meta` using neighborhood. Such as :code:`SimulatedAnnealing`.

    Examples
    --------
    >>> from zellij.core import PermutationVar
    >>> a = PermutationVar(10,"cities")
    >>> print(a)
    PermutationVar(cities, 10)
    >>> print(f"Label {a.label}, Length {len(a)}, Variable {a.variable}")
    Label cities, Length 10, Variable 10
    >>> a.random()
    [5, 8, 9, 3, 0, 6, 4, 7, 2, 1]
    >>> a.random(5)
    [[1, 3, 0, 2, 4, 6, 7, 5, 9, 8],
     [7, 8, 2, 4, 0, 9, 5, 6, 1, 3],
     [7, 4, 3, 2, 5, 9, 6, 0, 1, 8],
     [0, 8, 6, 1, 3, 5, 4, 7, 9, 2],
     [5, 3, 8, 9, 7, 2, 6, 1, 0, 4]]
    """

    def __init__(
        self,
        n: int,
        label: str,
        converter: Optional[PermutationConverter] = None,
        neighborhood: Optional[PermutationNeighborhood] = None,
    ):
        self.n = n
        self._index = 0
        super().__init__(label, converter, neighborhood)

    @Variable.converter.setter
    def converter(self, value: Optional[PermutationConverter]):
        self._converter_setter(value, PermutationConverter)

    @Variable.neighborhood.setter
    def neighborhood(self, value: Optional[PermutationNeighborhood]):
        self._neighborhood_setter(value, PermutationNeighborhood)

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, value: int):
        if value > 1:
            self._n = value
        else:
            raise InitializationError(f"In PermutationVar, n must be >1. Got {value}.")

    def random(self, size: Optional[int] = None) -> list:
        """random

        Parameters
        ----------
        size : int, optional
            Number of draws. If None, returns a single list.

        Returns
        -------
        out: list
            Return a single list if size is None, list of values from :code:`variables` else.
        """

        if size:
            res = np.tile(np.arange(self.n), (size, 1))
            for v in res:
                v = np.random.permutation(v)
            return res.tolist()
        else:
            return np.random.permutation(np.arange(self.n)).tolist()

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self) -> object:
        if self._index >= self.n:
            raise StopIteration
        res = _PermElem(f"{self.label}_{self._index}")
        self._index += 1
        return res

    def __len__(self):
        return self.n

    def __repr__(self):
        return f"{super().__repr__()}{self.n})"
