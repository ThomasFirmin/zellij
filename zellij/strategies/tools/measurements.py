# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.search_space import BaseFractal, Fractal
    from zellij.strategies.tools.geometry import Hypersphere, Direct

import logging

logger = logging.getLogger("zellij.direct_utils")


class Measurement(ABC):
    """Measurement

    Abstract class describing the measure
    of a fractal.

    """

    @abstractmethod
    def __call__(self, fractal: BaseFractal) -> float:
        pass


class Level(Measurement):
    """Level

    The level of the current fractal is use
    as a measure.

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.strategies.tools import Hypercube, Level

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Hypercube(a, measurement=Level())
    >>> print(sp.level, sp.measure)
    0 0.0
    >>> children = sp.create_children()
    >>> for c in children:
    ...     print(c.level, c.measure)
    1 1.0
    1 1.0
    1 1.0
    1 1.0
    """

    def __call__(self, fractal: BaseFractal) -> float:
        return float(fractal.level)


class Radius(Measurement):
    """Radius

    The radius of the current fractal is use
    as a measure.
    Fractals must have a :code:`radius`
    attribute.

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.strategies.tools import Hypersphere, Radius

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Hypersphere(a, measurement=Radius())
    >>> print(sp.level, sp.measure)
    0 0.5
    >>> children = sp.create_children()
    >>> for c in children:
    ...     print(c.level, c.measure)
    1 0.20710678118654754
    1 0.20710678118654754
    1 0.20710678118654754
    1 0.20710678118654754
    """

    def __call__(self, fractal: Hypersphere) -> float:
        return fractal.radius


class Direct_size(Measurement):
    """Direct_size

    Abstract class for Direct based measures.
    Fractals must have a :code:`upper` and
    :code:`lower` attributes.

    Returns:
        _type_: _description_
    """

    @abstractmethod
    def __call__(self, fractal: Direct) -> float:
        pass


class Sigma2(Direct_size):
    """Sigma2

    Computes the measure of a give hyperrectangle.
    Sigma function from locally biased DIRECT.
    """

    def __call__(self, fractal: Direct) -> float:
        n = fractal.size
        stage = n - len(fractal.set_i)

        return round(fractal.width * (n - 8 / 9 * stage) ** (1 / 2), 13)


class SigmaInf(Direct_size):
    """SigmaInf

    Computes the measure of a give hyperrectangle.
    Sigma infinite function from DIRECT.

    """

    def __call__(self, fractal: Direct) -> float:
        return round(fractal.width / 2, 15)
