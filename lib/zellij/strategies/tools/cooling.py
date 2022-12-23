# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:37:19+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import numpy as np
from abc import abstractmethod

import logging

logger = logging.getLogger("zellij.cooling")


class Cooling(object):
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

    def __init__(self, T0, Tend, peaks=1):

        ##############
        # PARAMETERS #
        ##############

        assert (
            T0 > Tend
        ), f"T0 must be stricly greater than Tend, got {T0}>{Tend}"
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

    """

    def __init__(self, alpha, T0, Tend, peaks=1):
        super().__init__(T0, Tend, peaks)
        self.alpha = alpha

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
            int(np.ceil(np.log(self.Tend / self.T0) / np.log(self.alpha)))
            * self.peaks
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

    """

    def __init__(self, alpha, T0, Tend, peaks=1):
        super().__init__(T0, Tend, peaks)
        self.alpha = alpha

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
            int(np.ceil(np.exp((self.T0 / self.Tend - 1 / self.alpha)) + 1))
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

    """

    def __init__(self, alpha, T0, Tend, peaks=1):
        super().__init__(T0, Tend, peaks)
        self.alpha = alpha

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

    """

    def __init__(self, alpha, T0, Tend, peaks=1):
        super().__init__(T0, Tend, peaks)
        self.alpha = alpha

    def cool(self):

        self.Tcurrent = self.T0 / (1 + self.alpha * self.k**2)

        if self.Tcurrent <= self.Tend:
            self.cross += 1
            self.k = 0
        else:
            self.k += 1

        return self.Tcurrent if self.cross < self.peaks else False

    def iterations(self):
        return (
            int(np.ceil(np.sqrt(self.T0 / (self.Tend * self.alpha))))
            * self.peaks
        )


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

    """

    def __init__(self, cycles, T0, Tend, peaks=1):
        super().__init__(T0, Tend, peaks)
        self.cycles = cycles

    def cool(self):

        self.Tcurrent = self.Tend + (self.T0 - self.Tend) * (
            (self.cycles - self.k) / self.cycles
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

    """

    def __init__(self, cycles, T0, Tend, peaks=1):
        super().__init__(T0, Tend, peaks)
        self.cycles = cycles

    def cool(self):

        self.Tcurrent = (
            self.Tend
            + (self.T0 - self.Tend)
            * ((self.cycles - self.k) / self.cycles) ** 2
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

    """

    def __init__(self, cycles, T0, Tend, peaks=1):
        super().__init__(T0, Tend, peaks)
        self.cycles = cycles

    def cool(self):

        self.Tcurrent = self.Tend + (self.T0 - self.Tend) * (
            1
            / (
                1
                + np.exp(
                    (2 * np.log(self.T0 - self.Tend) / self.cycles)
                    * (self.k - 0.5 * self.cycles)
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

    """

    def __init__(self, cycles, T0, Tend, peaks=1):
        super().__init__(T0, Tend, peaks)
        self.cycles = cycles

    def cool(self):

        self.Tcurrent = self.Tend + 0.5 * (self.T0 - self.Tend) * (
            1 + np.cos(self.k * np.pi / self.cycles)
        )

        if self.k == self.cycles:
            self.cross += 1
            self.k = 0
        else:
            self.k += 1

        return self.Tcurrent if self.cross < self.peaks else False

    def iterations(self):
        return self.cycles * self.peaks
