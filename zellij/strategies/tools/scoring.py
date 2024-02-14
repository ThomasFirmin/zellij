# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.search_space import BaseFractal, Fractal
    from zellij.core.loss_func import SequentialLoss

import numpy as np
import logging

logger = logging.getLogger("zellij.scoring")


class Scoring(ABC):
    """Scoring

    Scoring is an abstract class defining the scoring method of DAC.
    It is similar to an acquisition function in BO. According to sampled points,
    it gives a score to a :ref:`frac`, which determines how promising it is.

    """

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, fractal: BaseFractal) -> float:
        """__call__(fractal)

        Abstract method

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Minimal score found.

        """
        pass


class Min(Scoring):
    """Min

    Returns
    -------
    out : float
        Minimal score found inside the fractal

    Examples
    --------
    >>> from zellij.core import ArrayVar, FloatVar, Loss, Minimizer
    >>> from zellij.strategies.tools import Hypercube, Min
    >>> import numpy as np

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     x = np.array(x)*10-5
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Hypercube(a)
    >>> scoring = Min()
    >>> points = sp.random_point(100)
    >>> p,y,_,_ = himmelblau(points)
    >>> sp.add_solutions(p,y)
    >>> print(scoring(sp))
    0.9000984832813494
    """

    def __call__(self, fractal: BaseFractal) -> float:
        """__call__(fractal)

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Minimal score found.

        """
        if len(fractal.losses) > 0:
            return np.min(fractal.losses)
        else:
            return fractal.score


class Improvement(Scoring):
    """Improvement

    Returns
    -------
    out : float
        Improvement between min current score and father score.
    """

    def __call__(self, fractal: BaseFractal) -> float:
        """__call__(fractal)

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Minimal score found.

        """
        if len(fractal.losses) > 0:
            minl = np.min(fractal.losses)
            if np.isfinite(fractal.score):
                ok = minl - fractal.score
                return ok
            else:
                return minl
        else:
            return fractal.score


class Nothing(Scoring):
    """Nothing

    Does not modify current score.

    Returns
    -------
    out : float
        Return score of the current fractal.

    Examples
    --------
    >>> from zellij.core import ArrayVar, FloatVar, Loss, Minimizer
    >>> from zellij.strategies.tools import Hypercube, Nothing
    >>> import numpy as np

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     x = np.array(x)*10-5
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Hypercube(a)
    >>> sp.score = 10000
    >>> scoring = Nothing()
    >>> points = sp.random_point(100)
    >>> p,y,_,_ = himmelblau(points)
    >>> sp.add_solutions(p,y)
    >>> print(scoring(sp))
    10000

    """

    def __call__(self, fractal: BaseFractal) -> float:
        """__call__(fractal)

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Score of the fractal

        """
        return fractal.score


class Median(Scoring):
    """Median

    Returns
    -------
    out : float
        Median score found inside the fractal

    Examples
    --------
    >>> from zellij.core import ArrayVar, FloatVar, Loss, Minimizer
    >>> from zellij.strategies.tools import Hypercube, Median
    >>> import numpy as np

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     x = np.array(x)*10-5
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Hypercube(a)
    >>> scoring = Median()
    >>> points = sp.random_point(100)
    >>> p,y,_,_ = himmelblau(points)
    >>> sp.add_solutions(p,y)
    >>> print(scoring(sp))
    106.20458927202814
    """

    def __call__(self, fractal: BaseFractal) -> float:
        """__call__(fractal)

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Median score found.

        """
        if len(fractal.losses) > 0:
            return float(np.median(fractal.losses))
        else:
            return fractal.score


class Mean(Scoring):
    """Mean

    Returns
    -------
    out : float
        Mean score found inside the fractal

    Examples
    --------
    >>> from zellij.core import ArrayVar, FloatVar, Loss, Minimizer
    >>> from zellij.strategies.tools import Hypercube, Mean
    >>> import numpy as np

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     x = np.array(x)*10-5
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Hypercube(a)
    >>> scoring = Mean()
    >>> points = sp.random_point(100)
    >>> p,y,_,_ = himmelblau(points)
    >>> sp.add_solutions(p,y)
    >>> print(scoring(sp))
    133.97811550016144
    """

    def __call__(self, fractal: BaseFractal) -> float:
        """__call__(fractal)

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Mean score found.

        """
        if len(fractal.losses) > 0:
            return float(np.mean(fractal.losses))
        else:
            return fractal.score


class Std(Scoring):
    """Std

    Standard deviation

    Returns
    -------
    out : float
        Std score found inside the fractal

    Examples
    --------
    >>> from zellij.core import ArrayVar, FloatVar, Loss, Minimizer
    >>> from zellij.strategies.tools import Hypercube, Std
    >>> import numpy as np

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     x = np.array(x)*10-5
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Hypercube(a)
    >>> scoring = Std()
    >>> points = sp.random_point(100)
    >>> p,y,_,_ = himmelblau(points)
    >>> sp.add_solutions(p,y)
    >>> print(scoring(sp))
    105.03233132857676

    """

    def __call__(self, fractal: BaseFractal) -> float:
        """__call__(fractal)

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Standard deviation.

        """
        if len(fractal.losses) > 0:
            return float(np.std(fractal.losses))
        else:
            return fractal.score


class DistanceToTheBest(Scoring):
    """DistanceToTheBest

    Does not work with MPILoss

    Returns
    -------
    out : float
        DistanceToTheBest score found inside the fractal

    Examples
    --------
    >>> from zellij.core import ArrayVar, FloatVar, Loss, Minimizer
    >>> from zellij.strategies.tools import Hypercube, DistanceToTheBest
    >>> import numpy as np

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     x = np.array(x)*10-5
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Hypercube(a)
    >>> scoring = DistanceToTheBest(himmelblau)
    >>> points = sp.random_point(100)
    >>> p,y,_,_ = himmelblau(points)
    >>> sp.add_solutions(p,y)
    >>> print(scoring(sp))
    -6.27635704124182e+19
    """

    def __init__(self, loss: SequentialLoss):
        """__init__

        Parameters
        ----------
        loss : LossFunc
            :ref:`lf` necessary to have access to the best point found so far.
            Does not work with MPILoss
        """
        super().__init__()
        self.loss = loss

    def __call__(self, fractal: Fractal) -> float:
        """__call__(fractal)

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Distance to the best solution found so far.

        """
        if len(fractal.losses) > 0:
            best_ind = self.loss.best_point
            distances = [
                fractal.distance(s, best_ind) + 1e-20 for s in fractal.solutions
            ]
            res = -np.max(np.array(fractal.losses) / distances)
            return float(res)
        else:
            return fractal.score


class DistanceToTheBestCentered(Scoring):
    """DistanceToTheBestCentered

    Does not work with MPILoss

    Returns
    -------
    out : float
        Distance_to_the_best score found inside the fractal

    Examples
    --------
    >>> from zellij.core import ArrayVar, FloatVar, Loss, Minimizer
    >>> from zellij.strategies.tools import Hypercube, DistanceToTheBestCentered
    >>> import numpy as np

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     x = np.array(x)*10-5
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Hypercube(a)
    >>> scoring = DistanceToTheBestCentered(himmelblau)
    >>> points = sp.random_point(100)
    >>> p,y,_,_ = himmelblau(points)
    >>> sp.add_solutions(p,y)
    >>> print(scoring(sp))
    -1144.4597133177895

    """

    def __init__(self, loss: SequentialLoss):
        """__init__

        Parameters
        ----------
        loss : LossFunc
            :ref:`lf` necessary to have access to the best point found so far.
            Does not work with MPILoss
        """
        super().__init__()
        self.loss = loss

    def __call__(self, fractal: Fractal) -> float:
        """__call__(fractal)

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Distance to the best solution found so far.

        """
        if len(fractal.losses) > 0:
            best_ind = self.loss.best_point
            distances = [
                fractal.distance(s, best_ind) + 1e-20 for s in fractal.solutions
            ]
            res = -np.max((np.array(fractal.losses) - self.loss.best_score) / distances)
            return float(res)
        else:
            return fractal.score


class Belief(Scoring):
    """Belief

    Does not work with MPILoss

    Returns
    -------
    out : float
        Belief score found inside the fractal

    Examples
    --------
    >>> from zellij.core import ArrayVar, FloatVar, Loss, Minimizer
    >>> from zellij.strategies.tools import Hypercube, Belief
    >>> import numpy as np

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     x = np.array(x)*10-5
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}

    >>> a = ArrayVar(FloatVar("f1", 0, 1), FloatVar("i2", 0, 1))
    >>> sp = Hypercube(a)
    >>> scoring = Belief(himmelblau, gamma=0.3)
    >>> points = sp.random_point(100)
    >>> p,y,_,_ = himmelblau(points)
    >>> sp.add_solutions(p,y)
    >>> print(scoring(sp))
    -0.02536640307712946

    """

    def __init__(self, loss: SequentialLoss, gamma: float = 0.5):
        """__init__

        Parameters
        ----------
        loss : SequentialLoss
            :ref:`lf` necessary to have access to the best point found so far.
            Does not work with MPILoss
        gamma : float, default=0.5
            Influence of the parent score on the child.

        """
        super().__init__()
        self.loss = loss
        self.gamma = gamma

    def __call__(self, fractal: Fractal) -> float:
        """__call__

        Parameters
        ----------
        fractal : Fractal
            Fractal containing all solutions sampled within it,
            and their corresponding objective losses.

        Returns
        -------
        out : float
            Belief from FRACTOP.

        """
        best_sc = self.loss.best_score

        if len(fractal.losses) > 0:
            if np.isfinite(fractal.score):
                father_score = fractal.score
            else:
                father_score = 0

            ratio = np.array(fractal.losses) / best_sc
            # Negate because minimization problem and maximize Belief
            res = -(
                self.gamma * father_score
                + (1 - self.gamma) * np.mean(ratio * np.exp(1 - ratio))
            )
            return float(res)
        else:
            return fractal.score
