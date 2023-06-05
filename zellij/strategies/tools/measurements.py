# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-06-17T18:23:26+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-23T12:34:15+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
import numpy as np
from abc import ABC, abstractmethod

import logging

logger = logging.getLogger("zellij.direct_utils")


class Measurement(ABC):
    """Measurement

    Abstract class describing the measure
    of a fractal.

    """

    @abstractmethod
    def __call__(self, fractal):
        pass


class Level(Measurement):
    """Level

    The level of the current fractal is use
    as a measure.

    """

    def __call__(self, fractal):
        return fractal.level


class Radius(Measurement):
    """Radius

    The radius of the current fractal is use
    as a measure.
    Fractals must have a :code:`radius`
    attribute.

    """

    def __call__(self, fractal):
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
    def __call__(self, fractal):
        pass


class Sigma2(Direct_size):
    """Sigma2

    Computes the measure of a give hyperrectangle.
    Sigma function from locally biased DIRECT.
    """

    def __call__(self, fractal):
        upmlo = fractal.upper - fractal.lower
        n = len(upmlo)

        longest = np.max(upmlo)
        stage = n - len(upmlo == longest)
        return round(
            longest * (n - 8 / 9 * stage) ** (1 / 2),
            13,
        )


class SigmaInf(Direct_size):
    """SigmaInf

    Computes the measure of a give hyperrectangle.
    Sigma infinite function from DIRECT.

    """

    def __call__(self, fractal):
        upmlo = fractal.upper - fractal.lower
        longest = np.max(upmlo)
        return round(longest / 2, 13)
