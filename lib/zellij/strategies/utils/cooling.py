# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   ThomasFirmin
# @Last modified time: 2022-05-03T15:44:44+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


import numpy as np
from abc import abstractmethod
import matplotlib.pyplot as plt


class Cooling(object):
    """Cooling

    Cooling is a base object which define what a cooling Schedule is.

    Parameters
    ----------
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionally to\
        `T0`. It allows to periodically easily escape from local optima.

    peaks : int, default=1
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
        `peaks` times.

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
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
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

    def show(self, filepath=""):

        pts = [self.cool() for i in range(self.iterations())]
        self.reset()

        fig, ax = plt.subplots(figsize=(19.2, 14.4))
        fig.suptitle(f"{self.__class__.__name__} cooling schedule")
        plt.xlabel("Iterations")
        plt.ylabel("Temperature")
        plt.plot(pts, ls="-", color="orange")

        if filepath:
            save_path = os.path.join(
                self.loss_func.plots_path, f"cooling_sa.png"
            )

            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
            plt.close()


class MulExponential(Cooling):
    """MulExponential

    Exponential multiplicative monotonic cooling.

    :math:`T_k = T_0.\\alpha^k`

    Parameters
    ----------
    alpha : float
        Decrease factor. 0.8<=`alpha`<=0.9
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionally to\
        `T0`. It allows to periodically easily escape from local optima.

    peaks : int, default=1
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
        `peaks` times.

    Attributes
    ----------
    alpha

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

        self.Tcurrent = self.T0 * self.alpha ** self.k

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
        Decrease factor. `alpha`>1
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionally to\
        `T0`. It allows to periodically easily escape from local optima.

    peaks : int, default=1
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
        `peaks` times.

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
            int(np.ceil(np.exp((self.T0 / self.Tend - 1 / self.alpha)) - 1))
            * self.peaks
        )


class MulLinear(Cooling):
    """MulLinear

    Linear multiplicative monotonic cooling.

    :math:`T_k = \\frac{T_0}{1+\\alpha.k}`

    Parameters
    ----------
    alpha : float
        Decrease factor. `alpha`>0
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionally to\
        `T0`. It allows to periodically easily escape from local optima.

    peaks : int, default=1
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
        `peaks` times.

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
        Decrease factor. `alpha`>0
    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionally to\
        `T0`. It allows to periodically easily escape from local optima.

    peaks : int, default=1
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
        `peaks` times.

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

        self.Tcurrent = self.T0 / (1 + self.alpha * self.k ** 2)

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
        `T0`. It allows to periodically easily escape from local optima.

    peaks : int, default=1
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
        `peaks` times.

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

    Parameters
    ----------
    cycles : int
        Number of cooling cycles.

    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionally to\
        `T0`. It allows to periodically easily escape from local optima.

    peaks : int, default=1
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
        `peaks` times.

    Attributes
    ----------
    cycles

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

    Parameters
    ----------
    cycles : int
        Number of cooling cycles.

    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionally to\
        `T0`. It allows to periodically easily escape from local optima.

    peaks : int, default=1
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
        `peaks` times.

    Attributes
    ----------
    cycles

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
                    2
                    * np.log(self.T0 - self.Tend)
                    / self.cycles
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

    Parameters
    ----------
    cycles : int
        Number of cooling cycles.

    T0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    Tend : float
        Temperature threshold. When reached the temperature is violently increased proportionally to\
        `T0`. It allows to periodically easily escape from local optima.

    peaks : int, default=1
        Maximum number of crossed threshold according to `Tend`. The temperature will be increased\
        `peaks` times.

    Attributes
    ----------
    cycles

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
