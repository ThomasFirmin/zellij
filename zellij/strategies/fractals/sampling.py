# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from zellij.core.errors import InitializationError
from zellij.strategies.tools.chaos_map import ChaosMap
from zellij.core.search_space import (
    BaseFractal,
    Fractal,
    ContinuousSearchspace,
    DiscreteSearchspace,
    UnitSearchspace,
)
from zellij.core.metaheuristic import Metaheuristic

from zellij.strategies.tools import (
    Hypersphere,
    Section,
    NMSOSection,
    Hypercube,
    Direct,
    PermFractal,
)

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union

import numpy as np
import logging

logger = logging.getLogger("zellij.sampling")


class Sampling(Metaheuristic):

    @property
    def search_space(
        self,
    ) -> Union[BaseFractal, List[BaseFractal]]:
        if len(self._search_space) == 1:
            return self._search_space[0]
        else:
            return self._search_space

    @search_space.setter
    def search_space(self, value: Union[BaseFractal, List[BaseFractal]]):
        self.computed = False
        if isinstance(value, BaseFractal):
            self._search_space = [value]
        elif isinstance(value, list):
            self._search_space = value
        else:
            raise ValueError(f"Search space must be a list searchpaces, got {value}")


class ContinuousSampling(Sampling):
    """ContinuousSampling.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    See Also
    --------
    :ref:`lf` : Parent class for a loss function.
    :ref:`sp` : Defines what a search space is in Zellij.
    """

    def __init__(
        self, search_space: Union[ContinuousSearchspace, Fractal], verbose: bool = True
    ):
        super().__init__(search_space=search_space, verbose=verbose)
        self.xinfo = ["fracid"]

    @property
    def search_space(
        self,
    ) -> Union[
        ContinuousSearchspace, Fractal, List[Union[ContinuousSearchspace, Fractal]]
    ]:
        if len(self._search_space) == 1:
            return self._search_space[0]
        else:
            return self._search_space

    @search_space.setter
    def search_space(
        self,
        value: Union[
            ContinuousSearchspace, Fractal, List[Union[ContinuousSearchspace, Fractal]]
        ],
    ):
        self.computed = False
        if isinstance(value, (ContinuousSearchspace, Fractal)):
            self._search_space = [value]
        elif isinstance(value, list):
            self._search_space = value
        else:
            raise ValueError(
                f"Search space must be a list of continuous searchpaces, a fractals or have a `converter` addon, got {value}"
            )


class UnitSampling(Sampling):
    """UnitSampling

    Attributes
    ----------
    search_space : UnitSearchspace
        :ref:`sp` object containing decision variables and the loss function.

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    See Also
    --------
    :ref:`lf` : Parent class for a loss function.
    :ref:`sp` : Defines what a search space is in Zellij.
    """

    def __init__(self, search_space: UnitSearchspace, verbose: bool = True):
        super().__init__(search_space=search_space, verbose=verbose)

    @property
    def search_space(self) -> List[UnitSearchspace]:
        return self._search_space

    @search_space.setter
    def search_space(self, value: Union[UnitSearchspace, List[UnitSearchspace]]):
        if isinstance(value, UnitSearchspace):
            self._search_space = [value]
        elif isinstance(value, list):
            self._search_space = value
        else:
            raise ValueError(
                f"Search space must be a UnitSearchspace or a list of UnitSearchspace. Got {value}"
            )


class DiscreteSampling(Sampling):
    """DiscreteSampling

    ContinuousMetaheuristic is a subclass of :ref:`meta`, describing a
    metaheuristic working only with a discrete :ref:`sp`.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    See Also
    --------
    :ref:`lf` : Parent class for a loss function.
    :ref:`sp` : Defines what a search space is in Zellij.
    """

    def __init__(self, search_space: DiscreteSearchspace, verbose: bool = True):
        super().__init__(search_space=search_space, verbose=verbose)

    @property
    def search_space(self) -> List[DiscreteSearchspace]:
        return self._search_space

    @search_space.setter
    def search_space(
        self, value: Union[List[DiscreteSearchspace], DiscreteSearchspace]
    ):
        if isinstance(value, DiscreteSearchspace):
            self._search_space = [value]
        elif isinstance(value, list):
            self._search_space = value
        else:
            raise ValueError(
                "Search space must be discrete or have a `converter` addon"
            )


class Center(ContinuousSampling):
    """Center

    Samples the center of the targeted search space.
    The search space must have a :code:`center` attribute, or
    upper and lower bounds.

    Attributes
    ----------
    search_space : {Hyperpshere, Hypercube, Section}
            :ref:`sp` containing bounds of the search space.
    verbose : boolean, default=True
        Algorithm verbosity

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.strategies.tools import Hypersphere
    >>> from zellij.utils import ArrayDefaultC, FloatMinMax
    >>> from zellij.core import Experiment, Loss, Minimizer, Calls
    >>> from zellij.strategies.fractals import Center

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}

    >>> a = ArrayVar(
    ...     FloatVar("f1", -5, 5, converter=FloatMinMax()),
    ...     FloatVar("i2", -5, 5, converter=FloatMinMax()),
    ...     converter=ArrayDefaultC(),
    ... )
    >>> sp = Hypersphere(a)
    >>> opt = Center(sp)
    >>> stop = Calls(himmelblau, 1)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run()
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([0.0, 0.0])=170.0
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 1
    """

    def __init__(
        self, search_space: Union[Hypercube, Hypersphere, Section], verbose: bool = True
    ):
        """__init__

        Parameters
        ----------
        search_space : {Hyperpshere, Hypercube, Section}
            Search space object containing bounds of the search space.
        verbose : boolean, default=True
            Algorithm verbosity

        """
        super().__init__(search_space, verbose)
        self.computed = False

    def reset(self):
        super().reset()
        self.computed = False

    # RUN Center
    def forward(
        self,
        X: list,
        Y: np.ndarray,
        constraint: Optional[np.ndarray] = None,
        info: Optional[np.ndarray] = None,
        xinfo: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict, dict]:
        """
        Runs one step of Center.
        Center does not use secondary and constraint.

        Parameters
        ----------
        X : list
            List of points.
        Y : numpy.ndarray[float]
            List of loss values.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """
        if self.computed:
            return [], {"algorithm": "Center"}, {}
        else:

            pinfo = {"fracid": np.arange(0, len(self.search_space), dtype=int)}
            res = np.empty((len(self.search_space), self._search_space[0].size))
            for i, s in enumerate(self._search_space):
                res[i] = (s.upper + s.lower) / 2

            self.computed = True
            return res.tolist(), {"algorithm": "Center"}, pinfo


class Random(ContinuousSampling):
    """Random

    Samples the center of the targeted search space.
    The search space must have a :code:`center` attribute, or
    upper and lower bounds.

    Attributes
    ----------
    search_space : {Hyperpshere, Hypercube, Section}
            :ref:`sp` containing bounds of the search space.
    verbose : boolean, default=True
        Algorithm verbosity

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        search_space: Union[Hypercube, Hypersphere, Section],
        n: int,
        verbose: bool = True,
    ):
        """__init__

        Parameters
        ----------
        search_space : {Hyperpshere, Hypercube, Section}
            Search space object containing bounds of the search space.
        verbose : boolean, default=True
            Algorithm verbosity

        """
        super().__init__(search_space, verbose)
        self.n = n
        self.computed = False

    def reset(self):
        super().reset()
        self.computed = False

    # RUN Center
    def forward(
        self,
        X: list,
        Y: np.ndarray,
        constraint: Optional[np.ndarray] = None,
        info: Optional[np.ndarray] = None,
        xinfo: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict, dict]:
        """
        Runs one step of Center.
        Center does not use secondary and constraint.

        Parameters
        ----------
        X : list
            List of points.
        Y : numpy.ndarray[float]
            List of loss values.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """
        if self.computed:
            return [], {"algorithm": "Center"}, {}
        else:
            pinfo = {
                "fracid": np.repeat(
                    np.arange(0, len(self.search_space), dtype=int), self.n
                )
            }
            res = np.empty(
                (len(self.search_space) * self.n, self._search_space[0].size)
            )
            for i, s in enumerate(self._search_space):
                idx = i * self.n
                slc = slice(idx, idx + self.n)
                res[slc] = np.random.uniform(size=(self.n, s.size))
                res[slc] = s.lower + (s.upper - s.lower) * res[slc]

            self.computed = True
            return res.tolist(), {"algorithm": "Random"}, pinfo


# Promising Hypersphere Search
class PHS(ContinuousSampling):
    """PHS

    Promising Hypersphere Search  is an exploration algorithm comming from the original FDA paper.
    It is used to evaluate the center of an Hypersphere, and fixed points on each dimension arround this center.

    Attributes
    ----------
    search_space : Hypersphere
        A Hypersphere.
    inflation : float, default=1.75
        Inflation rate of the :code:`Hypersphere`
    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    Methods
    -------
    forward(X, Y)
        Runs one step of PHS.

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.strategies.tools import Hypersphere
    >>> from zellij.utils import ArrayDefaultC, FloatMinMax
    >>> from zellij.core import Experiment, Loss, Minimizer, Calls
    >>> from zellij.strategies.fractals import PHS

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}

    >>> a = ArrayVar(
    ...     FloatVar("f1", -5, 5, converter=FloatMinMax()),
    ...     FloatVar("i2", -5, 5, converter=FloatMinMax()),
    ...     converter=ArrayDefaultC(),
    ... )
    >>> sp = Hypersphere(a)
    >>> opt = PHS(sp, inflation=1)
    >>> stop = Calls(himmelblau, 3)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run()
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([-3.5355339059327373, -3.5355339059327373])=8.002525316941673
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 3
    """

    def __init__(
        self, search_space: Hypersphere, inflation: float = 1.75, verbose: bool = True
    ):
        """__init__

        Parameters
        ----------
        search_space : Hypersphere
            :ref:`sp` object containing decision variables and the loss function.
        inflation : float, default=1.75
            Inflation rate of the :code:`Hypersphere`
        verbose : boolean, default=True
            Activate or deactivate the progress bar.

        """

        self.inflation = inflation
        super().__init__(search_space, verbose)
        self.computed = False

    # RUN PHS
    def forward(
        self,
        X: list,
        Y: np.ndarray,
        constraint: Optional[np.ndarray] = None,
        info: Optional[np.ndarray] = None,
        xinfo: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict, dict]:
        """
        Runs one step of PHS.
        PHS does not use secondary and constraint.

        Parameters
        ----------
        X : list
            List of points.
        Y : numpy.ndarray[float]
            List of loss values.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """

        if self.computed:
            return [], {"algorithm": "EndPHS"}, {}
        else:

            pinfo = {
                "fracid": np.repeat(np.arange(0, len(self.search_space), dtype=int), 3)
            }
            res = []
            for s in self._search_space:
                if isinstance(s, Hypersphere):
                    points = np.tile(s.center, (3, 1))
                    points[1] -= s.radius * self.inflation
                    points[2] += s.radius * self.inflation
                    points[1:] = np.clip(points[1:], 0.0, 1.0)
                    res.extend(points.tolist())
                else:
                    raise ValueError(
                        f"Searchspace must be a list of Hyperpsheres, Sections, Hypercubes or LatinHypercubes, got {type(s)}."
                    )

            self.computed = True
            return res, {"algorithm": "Center"}, pinfo

    def reset(self):
        super().reset()
        self.computed = False


class CenterSOO(ContinuousSampling):
    """CenterSOO

    Samples the center of the targeted search space.
    The search space must be a Section.
    If fractal :code:`is_middle`, then return empty list.

    Attributes
    ----------
    search_space : Section
            :ref:`sp` containing bounds of the search space.
    verbose : boolean, default=True
        Algorithm verbosity


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.strategies.tools import Section
    >>> from zellij.utils import ArrayDefaultC, FloatMinMax
    >>> from zellij.core import Experiment, Loss, Minimizer, Calls
    >>> from zellij.strategies.fractals import CenterSOO

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}

    >>> a = ArrayVar(
    ...     FloatVar("f1", -5, 5, converter=FloatMinMax()),
    ...     FloatVar("i2", -5, 5, converter=FloatMinMax()),
    ...     converter=ArrayDefaultC(),
    ... )
    >>> sp = Section(a)
    >>> opt = CenterSOO(sp)
    >>> stop = Calls(himmelblau, 1)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run()
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([0.0, 0.0])=170.0
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 1

    """

    def __init__(self, search_space: Section, verbose: bool = True):
        """__init__

        Parameters
        ----------
        search_space : Section
            Section.
        verbose : boolean, default=True
            Algorithm verbosity

        """
        super().__init__(search_space, verbose)
        self.computed = False

    def forward(
        self,
        X: list,
        Y: np.ndarray,
        constraint: Optional[np.ndarray] = None,
        info: Optional[np.ndarray] = None,
        xinfo: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict, dict]:
        """
        Runs one step of CenterSOO.
        Center does not use secondary and constraint.

        Parameters
        ----------
        X : list
            List of points.
        Y : numpy.ndarray[float]
            List of loss values.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """
        if self.computed:
            return [], {"algorithm": "Center"}, {}
        else:

            fracid = []
            res = []
            for i, s in enumerate(self._search_space):
                if isinstance(s, (Section, NMSOSection)):
                    if (not s.is_middle and not s.evaluated) or s.level == 1:
                        center = (s.upper + s.lower) / 2
                        res.append(center.tolist())
                        fracid.append(i)
                else:
                    raise ValueError(
                        f"Searchspace must be a list of Hyperpsheres, Sections, Hypercubes or LatinHypercubes, got {type(s)}."
                    )

            pinfo = {"fracid": fracid}
            self.computed = True
            return res, {"algorithm": "CenterSOO"}, pinfo

    def reset(self):
        self.computed = False


class Diagonal(ContinuousSampling):
    """Diagonal

    Sample the center of the :ref:`sp`, and two equidistant points
    on the diagonal.
    The search space must be a :code:`Hypercube` or a :code:`Section`.

    Attributes
    ----------
    search_space : {Hypercube, Section}
            :ref:`sp` containing bounds of the search space.
    ratio : float, default=0.8
        0.0<:code:`ratio`<=1.0.
        Proportionnal distance of sampled points from the center.
    verbose : boolean, default=True
        Algorithm verbosity


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.strategies.tools import Section
    >>> from zellij.utils import ArrayDefaultC, FloatMinMax
    >>> from zellij.core import Experiment, Loss, Minimizer, Calls
    >>> from zellij.strategies.fractals import Diagonal

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}

    >>> a = ArrayVar(
    ...     FloatVar("f1", -5, 5, converter=FloatMinMax()),
    ...     FloatVar("i2", -5, 5, converter=FloatMinMax()),
    ...     converter=ArrayDefaultC(),
    ... )
    >>> sp = Section(a)
    >>> opt = Diagonal(sp)
    >>> stop = Calls(himmelblau, 3)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run()
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([-4.0, -4.0])=26.0
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 3

    """

    def __init__(
        self, search_space: Union[Hypercube, Section], ratio=0.8, verbose=True
    ):
        """__init__

        Parameters
        ----------
        search_space : {Hypercube, Section}
            :ref:`sp` containing bounds of the search space.
        ratio : float, default=0.8
            0.0<:code:`ratio`<=1.0.
            Proportionnal distance of sampled points from the center.
        verbose : boolean, default=True
            Algorithm verbosity

        """

        self.ratio = ratio
        super().__init__(search_space, verbose)
        self.computed = False

    @property
    def ratio(self) -> float:
        return self._ratio

    @ratio.setter
    def ratio(self, value: float):
        if value < 0 or value > 1:
            raise InitializationError(f"Ratio must be 0<{value}<1")
        else:
            self._ratio = value

    def forward(
        self,
        X: list,
        Y: np.ndarray,
        constraint: Optional[np.ndarray] = None,
        info: Optional[np.ndarray] = None,
        xinfo: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict, dict]:
        """
        Runs one step of Center.
        Center does not use secondary and constraint.

        Parameters
        ----------
        X : list
            List of points.
        Y : numpy.ndarray[float]
            List of loss values.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """

        if self.computed:
            return [], {"algorithm": "Diagonal"}, {}
        else:

            pinfo = {
                "fracid": np.repeat(np.arange(0, len(self.search_space), dtype=int), 3)
            }
            res = []
            for s in self._search_space:
                if isinstance(s, Hypersphere):
                    center = s.center
                    radius = center * self.ratio
                    points = np.tile(center, (3, 1))
                    points[1] += radius
                    points[2] -= radius
                    res.extend(points.tolist())
                elif isinstance(s, (Hypercube, Section, LatinHypercube)):
                    center = (s.upper + s.lower) / 2
                    radius = center * self.ratio
                    points = np.tile(center, (3, 1))
                    points[1] += radius
                    points[2] -= radius
                    res.extend(points.tolist())
                else:
                    raise ValueError(
                        f"Searchspace must be a list of Hyperpsheres, Sections, Hypercubes or LatinHypercubes, got {type(s)}."
                    )

            self.computed = True
            return res, {"algorithm": "Diagonal"}, pinfo


class ChaosSampling(ContinuousSampling):
    """ChaosSampling

    Sample chaotic points.

    Attributes
    ----------
    search_space : {Section, Hypercube}
    Section or Hypercube :ref:`sp`.
    verbose : boolean, default=True
    Algorithm verbosity

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij

    Examples
    --------
    >>> from zellij.core.variables import ArrayVar, FloatVar
    >>> from zellij.strategies.tools import Section
    >>> from zellij.utils import ArrayDefaultC, FloatMinMax
    >>> from zellij.core import Experiment, Loss, Minimizer, Calls
    >>> from zellij.strategies.fractals import ChaosSampling
    >>> from zellij.strategies.tools import Henon

    >>> @Loss(objective=Minimizer("obj"))
    >>> def himmelblau(x):
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     return {"obj": res}

    >>> a = ArrayVar(
    ...     FloatVar("f1", -5, 5, converter=FloatMinMax()),
    ...     FloatVar("i2", -5, 5, converter=FloatMinMax()),
    ...     converter=ArrayDefaultC(),
    ... )
    >>> sp = Section(a)
    >>> cmap = Henon(100, sp.size)
    >>> opt = ChaosSampling(sp, cmap)
    >>> stop = Calls(himmelblau, 100)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run()
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([-2.929577073910849, 3.2436841569767445])=1.0328083593187116
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 100
    """

    def __init__(
        self,
        search_space: Union[Section, Hypercube],
        cmap: ChaosMap,
        verbose: bool = True,
    ):
        """__init__

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        cmap : ChaosMap
            ChaosMap.
        verbose : boolean, default=True
            Algorithm verbosity

        """
        super().__init__(search_space, verbose)
        self.map = cmap
        self.computed = False

    def reset(self):
        super().reset()
        self.computed = False

    def forward(
        self,
        X: list,
        Y: np.ndarray,
        constraint: Optional[np.ndarray] = None,
        info: Optional[np.ndarray] = None,
        xinfo: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict, dict]:
        """
        Runs one step of Center.
        Center does not use secondary and constraint.

        Parameters
        ----------
        X : list
            List of points.
        Y : numpy.ndarray[float]
            List of loss values.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """

        if self.computed:
            return [], {"algorithm": "Chaos"}, {}
        else:

            pinfo = {
                "fracid": np.repeat(
                    np.arange(0, len(self.search_space), dtype=int), len(self.map.map)
                )
            }
            res = []
            for s in self._search_space:
                if isinstance(s, (Hypercube, Section, LatinHypercube)):
                    points = self.map.map * (s.upper - s.lower) + s.lower
                    res.append(points.tolist())
                else:
                    raise ValueError(
                        f"Searchspace must be a list of Sections, Hypercubes or LatinHypercubes, got {type(s)}."
                    )

            self.computed = True
            return res, {"algorithm": "Center"}, pinfo


# class ChaosHypersphere(ContinuousMetaheuristic):
#     """ChaosHypersphere

#     Sample chaotic points. Adapted to :code:`Hypersphere`.

#     Attributes
#      ----------
#     search_space : Hypersphere
#        Hypersphre :ref:`sp`.
#     verbose : boolean, default=True
#        Algorithm verbosity

#     See Also
#     --------
#     Metaheuristic : Parent class defining what a Metaheuristic is.
#     LossFunc : Describes what a loss function is in Zellij
#     Searchspace : Describes what a loss function is in Zellij

#     Examples
#     --------
#     >>> from zellij.core.variables import ArrayVar, FloatVar
#     >>> from zellij.strategies.tools import Hypersphere
#     >>> from zellij.utils import ArrayDefaultC, FloatMinMax
#     >>> from zellij.core import Experiment, Loss, Minimizer, Calls
#     >>> from zellij.strategies.fractals import ChaosHypersphere
#     >>> from zellij.strategies.tools import Henon

#     >>> @Loss(objective=Minimizer("obj"))
#     >>> def himmelblau(x):
#     ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
#     ...     return {"obj": res}

#     >>> a = ArrayVar(
#     ...     FloatVar("f1", -5, 5, converter=FloatMinMax()),
#     ...     FloatVar("i2", -5, 5, converter=FloatMinMax()),
#     ...     converter=ArrayDefaultC(),
#     ... )
#     >>> sp = Hypersphere(a)
#     >>> cmap = Henon(100, sp.size)
#     >>> opt = ChaosHypersphere(sp, cmap)
#     >>> stop = Calls(himmelblau, 100)
#     >>> exp = Experiment(opt, himmelblau, stop)
#     >>> exp.run()
#     >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
#     f([-3.8163175889483165, -3.362572915627342])=0.28135350231857104
#     >>> print(f"Calls: {himmelblau.calls}")
#     Calls: 100

#     """

#     def __init__(
#         self,
#         search_space: Hypersphere,
#         cmap: ChaosMap,
#         inflation: float = 1,
#         verbose: bool = True,
#     ):
#         """__init__

#         Parameters
#         ----------
#         search_space : Hypersphere
#             Hypersphere :ref:`sp`.
#         cmap : ChaosMap
#             ChaosMap.
#         inflation : float, default=1
#             Inflation of the Hypersphere.
#         verbose : boolean, default=True
#             Algorithm verbosity

#         """
#         super().__init__(search_space, verbose)
#         self.map = cmap
#         self.computed = False
#         self.inflation = inflation

#     @property
#     def inflation(self) -> float:
#         return self.inflation

#     @inflation.setter
#     def inflation(self, value: float):
#         if value <= 0:
#             raise InitializationError(f"In ChaosHypersphere, inflation must be > 0.")
#         else:
#             self._inflation = value

#     @ContinuousMetaheuristic.search_space.setter
#     def search_space(self, value: Hypersphere):
#         if value:
#             if isinstance(value, Hypersphere):
#                 self._search_space = value
#                 self.center = value.center
#                 self.radius = value.radius
#                 self.computed = False
#             else:
#                 raise InitializationError(
#                     f"Search space must be a Hyperpshere, Section, Hypercube, got {type(value)}."
#                 )
#         else:
#             raise InitializationError(
#                 f"Search space must be a Hyperpshere, Section, Hypercube, got {type(value)}."
#             )

#     def reset(self):
#         super().reset()
#         self.computed = False

#     def forward(
#         self,
#         X: list,
#         Y: np.ndarray,
#         constraint: Optional[np.ndarray] = None,
#         info: Optional[np.ndarray] = None,
#         xinfo: Optional[np.ndarray] = None,
#     ) -> Tuple[List[list], dict, dict]:
#         """
#         Runs one step of Center.
#         Center does not use secondary and constraint.

#         Parameters
#         ----------
#         X : list
#             List of points.
#         Y : numpy.ndarray[float]
#             List of loss values.

#         Returns
#         -------
#         points
#             Return a list of new points to be computed with the :ref:`lf`.
#         info
#             Additionnal information linked to :code:`points`

#         """

#         if self.computed:
#             return [], {"algorithm": "ChaosH"}, {}
#         else:
#             points = self.map.map
#             # Muller method
#             norm = np.linalg.norm(points, axis=1)[:, None]
#             radii = np.random.random((self.map.nvectors, 1)) ** (1 / self.map.params)
#             points = self.radius * (radii * points / norm) * self.center
#             points = np.clip(points, 0.0, 1.0)
#             self.computed = True
#             return points.tolist(), {"algorithm": "ChaosH"}, {}


class DirectSampling(ContinuousSampling):
    """DirectSampling

    Samples points as in DIRECT algorithms.

    Attributes
    ----------
    search_space : Direct
            Direct :ref:`sp`.
    verbose : boolean, default=True
        Algorithm verbosity

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(self, search_space: Direct, verbose: bool = True):
        """__init__(search_space, f_calls,verbose=True)

        Initialize PHS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(search_space, verbose)
        self.computed = False

    def reset(self):
        self.computed = False

    def forward(
        self,
        X: list,
        Y: np.ndarray,
        constraint: Optional[np.ndarray] = None,
        info: Optional[np.ndarray] = None,
        xinfo: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict, dict]:
        """
        Runs one step of Center.
        Center does not use secondary and constraint.

        Parameters
        ----------
        X : list
            List of points.
        Y : numpy.ndarray[float]
            List of loss values.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """

        if self.computed:
            return [], {"algorithm": "DIRECT"}, {}
        else:
            res = []
            fracid = []
            for i, s in enumerate(self._search_space):
                if isinstance(s, Direct):
                    center = (s.lower + s.upper) / 2
                    if s.level == 0:
                        n_points = 2 * len(s.set_i) + 1
                        start = 1
                    else:
                        n_points = 2 * len(s.set_i)
                        start = 0

                    fracid.append([i] * n_points)
                    section_length = s.width / 3

                    points = np.tile(center, (n_points, 1))

                    for i, p1, p2 in zip(
                        s.set_i,
                        points[start:-1:2],
                        points[start + 1 :: 2],
                    ):
                        p1[i] -= section_length
                        p2[i] += section_length

                    res.append(points.tolist())
                else:
                    raise InitializationError(
                        f"Search space must be a Direct fractal, got {type(s)}."
                    )

            self.computed = True
            return points.tolist(), {"algorithm": "DIRECT"}, {"fracid": fracid}


class Base(Metaheuristic):
    """Base

    Sample the base of a PermFractal.

    Attributes
    ----------
    search_space : PermFractal
        :ref:`sp` containing bounds of the search space.
    insertion : bool, default=False
        Insertion mode. Defines if the evaluation must be on the full permutation or
        only on the fixed elements.
    verbose : boolean, default=True
        Algorithm verbosity

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        search_space: Union[Hypercube, Hypersphere, Section],
        insertion=False,
        verbose: bool = True,
    ):
        """__init__

        Parameters
        ----------
        search_space : PermFractal
            Search space object containing bounds of the search space.
        insertion : bool, default=False
            Insertion mode. Defines if the evaluation must be on the full permutation or
            only on the fixed elements.
        verbose : boolean, default=True
            Algorithm verbosity

        """
        super().__init__(search_space, verbose)
        self.insertion = insertion
        self.computed = False

    @Metaheuristic.search_space.setter
    def search_space(self, value: PermFractal):
        if value:
            if isinstance(value, PermFractal):
                self._search_space = value
                self.base = value.base
                self.fixed_idx = value.fixed_idx
                self.computed = False
            else:
                raise InitializationError(
                    f"Search space must be a Hyperpshere, Section, Hypercube, got {type(value)}."
                )
        else:
            raise InitializationError(
                f"Search space must be a Hyperpshere, Section, Hypercube, got {type(value)}."
            )

    def reset(self):
        super().reset()
        self.computed = False

    # RUN Center
    def forward(
        self,
        X: list,
        Y: np.ndarray,
        constraint: Optional[np.ndarray] = None,
        info: Optional[np.ndarray] = None,
        xinfo: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict, dict]:
        """
        Runs one step of Center.
        Center does not use secondary and constraint.

        Parameters
        ----------
        X : list
            List of points.
        Y : numpy.ndarray[float]
            List of loss values.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """
        if self.computed:
            return [], {"algorithm": "Base"}, {}
        else:
            self.computed = True
            if self.insertion:
                res = np.full(len(self.base), -1, dtype=int)
                res[: self.fixed_idx] = self.base[: self.fixed_idx]
                return [res.tolist()], {"algorithm": "Base"}, {}
            else:
                return [self.base.tolist()], {"algorithm": "Base"}, {}
