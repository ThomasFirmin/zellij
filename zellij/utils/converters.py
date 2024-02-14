# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from zellij.core.addons import VarConverter
from zellij.core.variables import ArrayVar

from zellij.core.addons import (
    ArrayAddon,
    IntConverter,
    FloatConverter,
    CatConverter,
    ArrayConverter,
)
from zellij.core.errors import InitializationError, DimensionalityError

import numpy as np
import logging

logger = logging.getLogger("zellij.converters")
logger.setLevel(logging.INFO)

#########
# TOOLS #
#########


class Binning(VarConverter):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    @property
    def k(self) -> int:
        return self._k

    @k.setter
    def k(self, value: int):
        if isinstance(value, int) and value > 1:
            self._k = value
        else:
            raise InitializationError(
                f"k must be an int >1 for IntBinning, got {value}"
            )


####################################
# DO NOTHING
####################################


class DoNothing(VarConverter):
    """DoNothing

    :ref:`varadd` used when a :ref:`var` must not be converted.
    It does nothing, excep returning a non converted value.

    """

    def convert(self, value: object):
        return value

    def reverse(self, value: object):
        return value


####################################
# Int Converters
####################################


class IntMinMax(IntConverter):
    """IntMinmax

    Convert the value of an IntVar, using :math:`\\frac{x-lower}{upper-lower}=y`
    .Reverse: :math:`y(upper-lower)+lower=x`

    Examples
    --------
    >>> from zellij.core import IntVar
    >>> from zellij.utils import IntMinMax

    >>> a = IntVar("i1", 0, 255, converter=IntMinMax())
    >>> p = a.random()
    >>> intfloat = a.converter.convert(p)
    >>> floatint = a.converter.reverse(intfloat)
    >>> print(f"{p}->{intfloat:.1f}->{floatint}")
    240->0.9->240
    """

    def convert(self, value: int) -> float:
        return (value - self.target.lower) / (self.target.upper - self.target.lower)

    def reverse(self, value: float) -> int:
        return int(value * (self.target.upper - self.target.lower) + self.target.lower)


class IntBinning(Binning, IntConverter):
    """IntMinmax

    Convert a value from an IntVar using binning between its
    upper and lower bounds. Reversing a converted value will not return the
    initial value. When binning some information can be lost.

    Examples
    --------
    >>> from zellij.core import IntVar
    >>> from zellij.utils import IntBinning

    >>> a = IntVar("i1", 0, 1000, converter=IntBinning(11))
    >>> p = a.random()
    >>> intfloat = a.converter.convert(p)
    >>> floatint = a.converter.reverse(intfloat)
    >>> print(f"{p}->{intfloat}->{floatint}")
    847->8->800.0
    """

    def __init__(self, k: int):
        super(IntBinning, self).__init__(k=k)

    def convert(self, value: int) -> int:
        bins = np.linspace(self.target.lower, self.target.upper, self.k)
        return int(np.digitize(value, bins)) - 1

    def reverse(self, value: int) -> int:
        bins = np.linspace(self.target.lower, self.target.upper, self.k)
        return bins[value]


####################################
# Float Converters
####################################


class FloatMinMax(FloatConverter):
    """FloatMinmax

    Convert the value of a FloatVar, using
    :math:`\\frac{x-lower}{upper-lower}=y`
    .Reverse: :math:`y(upper-lower)+lower=x`

    Examples
    --------
    >>> from zellij.core import FloatVar
    >>> from zellij.utils import FloatMinMax

    >>> a = FloatVar("f1", -255, 255, converter=FloatMinMax())
    >>> p = a.random()
    >>> floatnorm = a.converter.convert(p)
    >>> normfloat = a.converter.reverse(floatnorm)
    >>> print(f"{p:.2f}->{floatnorm:.2f}->{normfloat:.2f}")
    231.61->0.95->231.61
    """

    def convert(self, value: float) -> float:
        return (value - self.target.lower) / (self.target.upper - self.target.lower)

    def reverse(self, value: float) -> float:
        return value * (self.target.upper - self.target.lower) + self.target.lower


class FloatBinning(Binning, FloatConverter):
    """FloatBinning

    Convert a value from an FloatVar using binning between its
    upper and lower bounds. Reversing a converted value will not return the
    initial value. When binning some information can be lost. here the decimal
    part of the float number.

    Examples
    --------
    >>> from zellij.core import FloatVar
    >>> from zellij.utils import FloatBinning

    >>> a = FloatVar("f1", 0, 1000, converter=FloatBinning(11))
    >>> p = a.random()
    >>> floatint = a.converter.convert(p)
    >>> intfloat = a.converter.reverse(floatint)
    >>> print(f"{p:.2f}->{floatnorm}->{intfloat:.2f}")
    646.27->6->600.00

    """

    def __init__(self, k: int):
        super(FloatBinning, self).__init__(k=k)

    def convert(self, value: float) -> int:
        bins = np.linspace(self.target.lower, self.target.upper, self.k)
        return int(np.digitize(value, bins)) - 1

    def reverse(self, value: int) -> float:
        bins = np.linspace(self.target.lower, self.target.upper, self.k)
        return bins[value]


####################################
# Categorical Converters
####################################


class CatToFloat(CatConverter):
    """CatMinmax

    Convert the value of a CatVar, using the index of the value in the list
    of the features of CatVar. :math:`\\frac{index}{len(features)}=y`
    .Reverse: :math:`features[floor(y*(len(features)-1))]=x`.

    Examples
    --------
    >>> from zellij.core import CatVar
    >>> from zellij.utils import CatToFloat

    >>> a = CatVar("c1", ["a", "b", "c"], converter=CatToFloat())
    >>> p = a.random()
    >>> catfloat = a.converter.convert(p)
    >>> floatcat = a.converter.reverse(catfloat)
    >>> print(f"{p}->{catfloat:.2f}->{floatcat}")
    b->0.33->b
    """

    def convert(self, value: object) -> float:
        return self.target.features.index(value) / len(self.target.features)

    def reverse(self, value: float) -> object:
        idx = int(value * len(self.target.features))
        if idx == len(self.target.features):
            idx -= 1

        return self.target.features[idx]


class CatToInt(CatConverter):
    """CatMinmax

    Convert the value of a CatVar to its corresponding index in the
    lsit of features.

    Examples
    --------
    >>> from zellij.core import CatVar
    >>> from zellij.utils import CatToInt

    >>> a = CatVar("c1", ["a", "b", "c"], converter=CatToInt())
    >>> p = a.random()
    >>> catint = a.converter.convert(p)
    >>> intcat = a.converter.reverse(catint)
    >>> print(f"{p}->{catint}->{intcat}")
    c->2->c
    """

    def convert(self, value: object) -> int:
        return self.target.features.index(value)

    def reverse(self, value: int) -> object:
        return self.target.features[value]


####################################
# Array Converters
####################################


class ArrayDefaultC(ArrayConverter):
    """ArrayDefault

    Default converter. Convert :ref:`var` one by one, by using their own converters.

    Examples
    --------
    >>> from zellij.core import ArrayVar, FloatVar, IntVar, CatVar
    >>> from zellij.utils import ArrayDefaultC, FloatMinMax, IntMinMax, CatToFloat

    >>> a = ArrayVar(
    ...     IntVar("i1", 0, 8, converter=IntMinMax()),
    ...     FloatVar("f1", 10, 20, converter=FloatMinMax()),
    ...     CatVar("c1", ["Hello", 87, 2.56], converter=CatToFloat()),
    ...     converter=ArrayDefaultC(),
    ... )

    >>> p = a.random()
    >>> tofloat = a.converter.convert(p)
    >>> top = a.converter.reverse(tofloat)
    >>> print(f"{p}->\n{tofloat}->\n{top}")
    [8, 15.10240466469826, 'Hello']->
    [1.0, 0.5102404664698261, 0.0]->
    [8, 15.10240466469826, 'Hello']
    """

    @ArrayAddon.target.setter
    def target(self, value: ArrayVar):
        if not isinstance(value, ArrayVar):
            raise InitializationError(
                f"The target value cannot be {type(value)}. ArrayVar expected."
            )
        elif any(v.converter is None for v in value.variables):
            raise InitializationError(
                f"If an ArrayVar has a converter, all variables within an ArrayVar must have a converter."
            )
        else:
            self._target = value

    def convert(self, value: list) -> list:
        res = []
        if len(self.target) == len(value):
            for var, v in zip(self.target.variables, value):
                res.append(var.converter.convert(v))  # type: ignore
        else:
            raise DimensionalityError(
                f"Array of variables does not have the same length as given value. Got {len(self.target)}={len(value)}."
            )

        return res

    def reverse(self, value: list) -> list:
        res = []
        if len(self.target) == len(value):
            for var, v in zip(self.target.variables, value):
                res.append(var.converter.reverse(v))  # type: ignore
        else:
            raise DimensionalityError(
                f"Array of variables does not have the same length as given value. Got {len(self.target)}={len(value)}."
            )

        return res
