# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod

from zellij.core.errors import InitializationError

import logging

logger = logging.getLogger("zellij.cooling")


class Cooling(ABC):
    """Cooling

    Cooling is a base object which defines what a cooling Schedule is.

    Attributes
    ----------
    Tcurrent : float
        Current temperature
    cross : int
        Count the number of times Tend is crossed.
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)
    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionally to\
        <T0>. It allows to periodically easily escape from local optima.
    peaks : int, default=1
        Maximum number of crossed threshold according to :code:`Tend`. The temperature will be increased\
        <peaks> times.

    Methods
    -------
    cool()
        Decrease temperature and return the current temperature.
    reset()
        Reset cooling schedule
    iterations()
        Get the theoretical number of iterations to end the schedule.

    """

    def __init__(self, T0: float, Tend: float, peaks: int = 1):
        ##############
        # PARAMETERS #
        ##############

        if T0 <= Tend:
            raise InitializationError(
                f"T0 must be stricly greater than Tend, got {T0}>{Tend}."
            )
        self.T0 = T0
        self.Tend = Tend
        self.peaks = peaks

        #############
        # VARIABLES #
        #############
        self.Tcurrent = self.T0
        self.k = 0
        self.cross = 0

    @abstractmethod
    def cool(self):
        pass

    @abstractmethod
    def iterations(self):
        pass

    def reset(self):
        self.Tcurrent = self.T0
        self.k = 0
        self.cross = 0


class MulExponential(Cooling):
    """MulExponential

    Exponential multiplicative monotonic cooling.

    :math:`T_k = T_0.\\alpha^k`

    Attributes
    ----------
    alpha : float
        Decrease factor. :math:`0.8 \\leq \\alpha \\leq 0.9`
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)
    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionally to\
        :code:`T0`. It allows to periodically easily escape from local optima.
    peaks : int, default=1
        Maximum number of crossed threshold according to :code:`Tend`. The temperature will be increased\
        :code:`peaks` times.

    Methods
    -------
    cool()
        Decrease temperature and return the current temperature.
    reset()
        Reset cooling schedule
    iterations()
        Get the theoretical number of iterations to end the schedule.

    Examples
    --------
    >>> from zellij.strategies.tools import MulExponential

    >>> cooling = MulExponential(0.83, 2, 1)
    >>> print(cooling.iterations())
    4
    >>> print(f"{cooling.T0}, {cooling.Tend}, {cooling.peaks}")
    10, 1, 1
    >>> temperature = cooling.cool()
    >>> while temperature:
    ...     print(f"{temperature:.1f}", end=", ")
    ...     temperature = cooling.cool()
    2.0, 1.7, 1.4, 1.1, 
    
    """

    def __init__(self, alpha: float, T0: float, Tend: float, peaks: int = 1):
        super().__init__(T0, Tend, peaks)
        self.alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        if value >= 0.8 and value <= 0.9:
            self._alpha = value
        else:
            raise InitializationError(f"alpha must be 0.8<=alpha<=0.9. Got {value}")

    def cool(self):
        self.Tcurrent = self.T0 * self.alpha**self.k

        if self.Tcurrent <= self.Tend:
            self.cross += 1
            self.k = 0
        else:
            self.k += 1

        return self.Tcurrent if self.cross < self.peaks else False

    def iterations(self):
        return (
            int(np.ceil(np.log(self.Tend / self.T0) / np.log(self.alpha))) * self.peaks
        )


class MulLogarithmic(Cooling):
    """MulLogarithmic

    Logarithmic multiplicative monotonic cooling.

    :math:`T_k = \\frac{T_0}{1+\\alpha.log(1+k)}`

    Parameters
    ----------
    alpha : float
        Decrease factor. :math:`\\alpha>1`
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)
    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionally to\
        :code:`T0`. It allows to periodically easily escape from local optima.
    peaks : int, default=1
        Maximum number of crossed threshold according to :code:`Tend`. The temperature will be increased\
        :code:`peaks` times.

    Methods
    -------
    cool()
        Decrease temperature and return the current temperature.
    reset()
        Reset cooling schedule
    iterations()
        Get the theoretical number of iterations to end the schedule.

    Examples
    --------
    >>> from zellij.strategies.tools import MulLogarithmic

    >>> cooling = MulLogarithmic(6, 10, 1)
    >>> print(cooling.iterations())
    4
    >>> print(f"{cooling.T0}, {cooling.Tend}, {cooling.peaks}")
    10, 1, 1
    >>> temperature = cooling.cool()
    >>> while temperature:
    ...     print(f"{temperature:.1f}", end=", ")
    ...     temperature = cooling.cool()
    10.0, 1.9, 1.3, 1.1,
    
    """

    def __init__(self, alpha: float, T0: float, Tend: float, peaks: int = 1):
        super().__init__(T0, Tend, peaks)
        self.alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        if value > 1:
            self._alpha = value
        else:
            raise InitializationError(f"alpha must be >1. Got {value}")

    def cool(self):
        self.Tcurrent = self.T0 / (1 + self.alpha * np.log(1 + self.k))

        if self.Tcurrent <= self.Tend:
            self.cross += 1
            self.k = 0
        else:
            self.k += 1

        return self.Tcurrent if self.cross < self.peaks else False

    def iterations(self):
        return (
            int(
                np.ceil(
                    np.exp((self.T0 / (self.alpha * self.Tend) - 1 / self.alpha)) - 1
                )
            )
            * self.peaks
        )


class MulLinear(Cooling):
    """MulLinear

    Linear multiplicative monotonic cooling.

    :math:`T_k = \\frac{T_0}{1+\\alpha.k}`

    Parameters
    ----------
    alpha : float
        Decrease factor. :math:`\\alpha>0`
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)
    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionally to\
        :code:`T0`. It allows to periodically easily escape from local optima.
    peaks : int, default=1
        Maximum number of crossed threshold according to :code:`Tend`. The temperature will be increased\
        :code:`peaks` times.

    Methods
    -------
    cool()
        Decrease temperature and return the current temperature.
    reset()
        Reset cooling schedule
    iterations()
        Get the theoretical number of iterations to end the schedule.

    Examples
    --------
    >>> from zellij.strategies.tools import MulLinear

    >>> cooling = MulLinear(2, 10, 1)
    >>> print(cooling.iterations())
    5
    >>> print(f"{cooling.T0}, {cooling.Tend}, {cooling.peaks}")
    10, 1, 1
    >>> temperature = cooling.cool()
    >>> while temperature:
    ...     print(f"{temperature:.1f}", end=", ")
    ...     temperature = cooling.cool()
    10.0, 3.3, 2.0, 1.4, 1.1,

    """

    def __init__(self, alpha: float, T0: float, Tend: float, peaks: int = 1):
        super().__init__(T0, Tend, peaks)
        self.alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        if value > 0:
            self._alpha = value
        else:
            raise InitializationError(f"alpha must be >0. Got {value}")

    def cool(self):
        self.Tcurrent = self.T0 / (1 + self.alpha * self.k)

        if self.Tcurrent <= self.Tend:
            self.cross += 1
            self.k = 0
        else:
            self.k += 1

        return self.Tcurrent if self.cross < self.peaks else False

    def iterations(self):
        return int(np.ceil(self.T0 / (self.Tend * self.alpha))) * self.peaks


class MulQuadratic(Cooling):
    """MulQuadratic

    Quadratic multiplicative monotonic cooling.

    :math:`T_k = \\frac{T_0}{1+\\alpha.k^2}`

    Parameters
    ----------
    alpha : float
        Decrease factor. :math:`\\alpha>0`
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)
    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionally to\
        :code:`T0`. It allows to periodically easily escape from local optima.
    peaks : int, default=1
        Maximum number of crossed threshold according to :code:`Tend`. The temperature will be increased\
        :code:`peaks` times.

    Methods
    -------
    cool()
        Decrease temperature and return the current temperature.
    reset()
        Reset cooling schedule
    iterations()
        Get the theoretical number of iterations to end the schedule.

    Examples
    --------
    >>> from zellij.strategies.tools import MulQuadratic

    >>> cooling = MulQuadratic(2, 10, 1)
    >>> print(cooling.iterations())
    5
    >>> print(f"{cooling.T0}, {cooling.Tend}, {cooling.peaks}")
    10, 1, 1
    >>> temperature = cooling.cool()
    >>> while temperature:
    ...     print(f"{temperature:.1f}", end=", ")
    ...     temperature = cooling.cool()
    10.0, 3.3, 2.0, 1.4, 1.1,
    
    """

    def __init__(self, alpha: float, T0: float, Tend: float, peaks: int = 1):
        super().__init__(T0, Tend, peaks)
        self.alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        if value > 0:
            self._alpha = value
        else:
            raise InitializationError(f"alpha must be >0. Got {value}")

    def cool(self):
        self.Tcurrent = self.T0 / (1 + self.alpha * self.k**2)

        if self.Tcurrent <= self.Tend:
            self.cross += 1
            self.k = 0
        else:
            self.k += 1

        return self.Tcurrent if self.cross < self.peaks else False

    def iterations(self):
        return int(np.ceil(np.sqrt(self.T0 / (self.Tend * self.alpha)))) * self.peaks


class AddLinear(Cooling):
    """AddLinear

    Linear additive monotonic cooling.

    :math:`T_k = T_{end} + (T_0-T_{end})\\left(\\frac{cycles-k}{cycles}\\right)`

    Parameters
    ----------
    cycles : int
        Number of cooling cycles.
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)
    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionally to\
        :code:`T0`. It allows to periodically easily escape from local optima.
    peaks : int, default=1
        Maximum number of crossed threshold according to :code:`Tend`. The temperature will be increased\
        :code:`peaks` times.

    Methods
    -------
    cool()
        Decrease temperature and return the current temperature.
    reset()
        Reset cooling schedule
    iterations()
        Get the theoretical number of iterations to end the schedule.

    Examples
    --------
    >>> from zellij.strategies.tools import AddLinear

    >>> cooling = AddLinear(4, 10, 1)
    >>> print(cooling.iterations())
    4
    >>> print(f"{cooling.T0}, {cooling.Tend}, {cooling.peaks}")
    10, 1, 1
    >>> temperature = cooling.cool()
    >>> while temperature:
    ...     print(f"{temperature:.1f}", end=", ")
    ...     temperature = cooling.cool()
    10.0, 7.0, 4.0, 1.0,
    
    """

    def __init__(self, cycles, T0, Tend, peaks=1):
        super().__init__(T0, Tend, peaks)
        self.cycles = cycles
        self._c = cycles - 1

    def cool(self):
        self.Tcurrent = self.Tend + (self.T0 - self.Tend) * (
            (self._c - self.k) / self._c
        )

        if self.k == self.cycles:
            self.cross += 1
            self.k = 0
        else:
            self.k += 1

        return self.Tcurrent if self.cross < self.peaks else False

    def iterations(self):
        return self.cycles * self.peaks


class AddQuadratic(Cooling):
    """AddQuadratic

    Quadratic additive monotonic cooling.

    :math:`T_k = T_{end} + (T_0-T_{end})\\left(\\frac{cycles-k}{cycles}\\right)^2`

    Attributes
    ----------
    cycles : int
        Number of cooling cycles.
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)
    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionally to\
        :code:`T0`. It allows to periodically easily escape from local optima.
    peaks : int, default=1
        Maximum number of crossed threshold according to :code:`Tend`. The temperature will be increased\
        :code:`peaks` times.

    Methods
    -------
    cool()
        Decrease temperature and return the current temperature.
    reset()
        Reset cooling schedule
    iterations()
        Get the theoretical number of iterations to end the schedule.

    Examples
    --------
    >>> from zellij.strategies.tools import AddQuadratic

    >>> cooling = AddQuadratic(4, 10, 1)
    >>> print(cooling.iterations())
    4
    >>> print(f"{cooling.T0}, {cooling.Tend}, {cooling.peaks}")
    10, 1, 1
    >>> temperature = cooling.cool()
    >>> while temperature:
    ...     print(f"{temperature:.1f}", end=", ")
    ...     temperature = cooling.cool()
    10.0, 5.0, 2.0, 1.0, 
    
    """

    def __init__(self, cycles: int, T0: float, Tend: float, peaks: int = 1):
        super().__init__(T0, Tend, peaks)
        self.cycles = cycles
        self._c = cycles - 1

    def cool(self):
        self.Tcurrent = (
            self.Tend + (self.T0 - self.Tend) * ((self._c - self.k) / self._c) ** 2
        )

        if self.k == self.cycles:
            self.cross += 1
            self.k = 0
        else:
            self.k += 1

        return self.Tcurrent if self.cross < self.peaks else False

    def iterations(self):
        return self.cycles * self.peaks


class AddExponential(Cooling):
    """AddExponential

    Exponential additive monotonic cooling.

    :math:`T_k = T_{end} + \\frac{T_0-T_{end}}{1+e^{\\frac{2ln(T_0-T_{end})}{cycles}}(k-0,5cycles)}`

    Attributes
    ----------
    cycles : int
        Number of cooling cycles.
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)
    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionally to\
        :code:`T0`. It allows to periodically easily escape from local optima.
    peaks : int, default=1
        Maximum number of crossed threshold according to :code:`Tend`. The temperature will be increased\
        :code:`peaks` times.

    Methods
    -------
    cool()
        Decrease temperature and return the current temperature.
    reset()
        Reset cooling schedule
    iterations()
        Get the theoretical number of iterations to end the schedule.

    Examples
    --------
    >>> from zellij.strategies.tools import AddExponential

    >>> cooling = AddExponential(4, 10, 1)
    >>> print(cooling.iterations())
    4
    >>> print(f"{cooling.T0}, {cooling.Tend}, {cooling.peaks}")
    10, 1, 1
    >>> temperature = cooling.cool()
    >>> while temperature:
    ...     print(f"{temperature:.1f}", end=", ")
    ...     temperature = cooling.cool()
    9.1, 7.1, 3.9, 1.9, 
    
    """

    def __init__(self, cycles: int, T0: float, Tend: float, peaks: int = 1):
        super().__init__(T0, Tend, peaks)
        self.cycles = cycles
        self._c = cycles - 1

    def cool(self):
        self.Tcurrent = self.Tend + (self.T0 - self.Tend) * (
            1
            / (
                1
                + np.exp(
                    (2 * np.log(self.T0 - self.Tend) / self._c)
                    * (self.k - 0.5 * self._c)
                )
            )
        )

        if self.k == self.cycles:
            self.cross += 1
            self.k = 0
        else:
            self.k += 1

        return self.Tcurrent if self.cross < self.peaks else False

    def iterations(self):
        return self.cycles * self.peaks


class AddTrigonometric(Cooling):
    """AddTrigonometric

    Trigonometric additive monotonic cooling.

    :math:`T_k = T_{end} + 0,5(T_0-T_{end})(1+cos(\\frac{k.\\pi}{cycles}))`

    Attributes
    ----------
    cycles : int
        Number of cooling cycles.
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)
    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionally to\
        :code:`T0`. It allows to periodically easily escape from local optima.
    peaks : int, default=1
        Maximum number of crossed threshold according to :code:`Tend`. The temperature will be increased\
        :code:`peaks` times.

    Methods
    -------
    cool()
        Decrease temperature and return the current temperature.
    reset()
        Reset cooling schedule
    iterations()
        Get the theoretical number of iterations to end the schedule.

    Examples
    --------
    >>> from zellij.strategies.tools import AddExponential

    >>> cooling = AddExponential(4, 10, 1)
    >>> print(cooling.iterations())
    4
    >>> print(f"{cooling.T0}, {cooling.Tend}, {cooling.peaks}")
    10, 1, 1
    >>> temperature = cooling.cool()
    >>> while temperature:
    ...     print(f"{temperature:.1f}", end=", ")
    ...     temperature = cooling.cool()
    10.0, 7.8, 3.3, 1.0, 
    
    """

    def __init__(self, cycles: int, T0: float, Tend: float, peaks: int = 1):
        super().__init__(T0, Tend, peaks)
        self.cycles = cycles
        self._c = cycles - 1

    def cool(self):
        self.Tcurrent = self.Tend + 0.5 * (self.T0 - self.Tend) * (
            1 + np.cos(self.k * np.pi / self._c)
        )

        if self.k == self.cycles:
            self.cross += 1
            self.k = 0
        else:
            self.k += 1

        return self.Tcurrent if self.cross < self.peaks else False

    def iterations(self):
        return self.cycles * self.peaks
