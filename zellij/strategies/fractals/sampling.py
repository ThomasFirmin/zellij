# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from zellij.core.errors import InitializationError
from zellij.strategies.tools.chaos_map import ChaosMap
from zellij.core.metaheuristic import ContinuousMetaheuristic, Metaheuristic
from zellij.strategies.tools import Hypersphere, Section, Hypercube, Direct, PermFractal

from typing import List, Tuple, Optional, Union

import numpy as np
import logging

logger = logging.getLogger("zellij.sampling")


class Center(ContinuousMetaheuristic):

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

    @ContinuousMetaheuristic.search_space.setter
    def search_space(self, value: Union[Hypercube, Hypersphere, Section]):
        if value:
            if isinstance(value, Hypersphere):
                self._search_space = value
                self.center = value.center
                self.computed = False
            elif isinstance(value, (Hypercube, Section)):
                self._search_space = value
                self.center = (value.upper + value.lower) / 2
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
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
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
            return [], {"algorithm": "Center"}
        else:
            self.computed = True
            return [self.center.tolist()], {"algorithm": "Center"}


class CenterSOO(ContinuousMetaheuristic):

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

    @property
    def search_space(self) -> Section:
        return self._search_space

    @search_space.setter
    def search_space(self, value: Section):
        if value and isinstance(value, Section):
            self._search_space = value
            self.center = (value.upper + value.lower) / 2
            self.computed = False
        else:
            raise InitializationError(
                f"Search space must be a Section, got {type(value)}."
            )

    def forward(
        self,
        X: list,
        Y: np.ndarray,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
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
        if self.computed or (
            np.isfinite(self.search_space.score) and self.search_space.is_middle
        ):
            self.computed = True
            return [], {"algorithm": "Center"}
        else:
            self.computed = True
            return [self.center.tolist()], {"algorithm": "Center"}

    def reset(self):
        self.computed = False


class Diagonal(ContinuousMetaheuristic):

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

    @ContinuousMetaheuristic.search_space.setter
    def search_space(self, value: Union[Hypercube, Section]):
        if value and isinstance(value, (Section, Hypercube)):
            self._search_space = value
            self.center = (value.upper + value.lower) / 2
            self.radius = self.center * self.ratio
            self.computed = False
        else:
            raise InitializationError(
                f"Search space must be a Section or Hypercube, got {type(value)}."
            )

    def forward(
        self,
        X: list,
        Y: np.ndarray,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
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
            return [], {"algorithm": "Diagonal"}
        else:
            points = np.tile(self.center, (3, 1))
            points[1] += self.radius
            points[2] -= self.radius
            self.computed = True
            return points.tolist(), {"algorithm": "Diagonal"}


class ChaosSampling(ContinuousMetaheuristic):

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

    @ContinuousMetaheuristic.search_space.setter
    def search_space(self, value: Union[Hypercube, Section]):
        if value and isinstance(value, (Section, Hypercube)):
            self._search_space = value
            self.computed = False
        else:
            raise InitializationError(
                f"Search space must be a Section or Hypercube, got {type(value)}."
            )

    def reset(self):
        super().reset()
        self.computed = False

    def forward(
        self,
        X: list,
        Y: np.ndarray,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
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
            return [], {"algorithm": "Chaos"}
        else:
            points = (
                self.map.map * (self.search_space.upper - self.search_space.lower)
                + self.search_space.lower
            )

        return points.tolist(), {"algorithm": "Chaos"}


class ChaosHypersphere(ContinuousMetaheuristic):

    """ChaosHypersphere

    Sample chaotic points. Adapted to :code:`Hypersphere`.

    Attributes
     ----------
    search_space : Hypersphere
       Hypersphre :ref:`sp`.
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
    >>> from zellij.strategies.fractals import ChaosHypersphere
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
    >>> sp = Hypersphere(a)
    >>> cmap = Henon(100, sp.size)
    >>> opt = ChaosHypersphere(sp, cmap)
    >>> stop = Calls(himmelblau, 100)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run()
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([-3.8163175889483165, -3.362572915627342])=0.28135350231857104
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 100

    """

    def __init__(
        self,
        search_space: Hypersphere,
        cmap: ChaosMap,
        inflation: float = 1,
        verbose: bool = True,
    ):
        """__init__

        Parameters
        ----------
        search_space : Hypersphere
            Hypersphere :ref:`sp`.
        cmap : ChaosMap
            ChaosMap.
        inflation : float, default=1
            Inflation of the Hypersphere.
        verbose : boolean, default=True
            Algorithm verbosity

        """
        super().__init__(search_space, verbose)
        self.map = cmap
        self.computed = False
        self.inflation = inflation

    @property
    def inflation(self) -> float:
        return self.inflation

    @inflation.setter
    def inflation(self, value: float):
        if value <= 0:
            raise InitializationError(f"In ChaosHypersphere, inflation must be > 0.")
        else:
            self._inflation = value

    @ContinuousMetaheuristic.search_space.setter
    def search_space(self, value: Hypersphere):
        if value:
            if isinstance(value, Hypersphere):
                self._search_space = value
                self.center = value.center
                self.radius = value.radius
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

    def forward(
        self,
        X: list,
        Y: np.ndarray,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
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
            return [], {"algorithm": "ChaosH"}
        else:
            points = self.map.map
            # Muller method
            norm = np.linalg.norm(points, axis=1)[:, None]
            radii = np.random.random((self.map.nvectors, 1)) ** (1 / self.map.params)
            points = self.radius * (radii * points / norm) * self.center
            points = np.clip(points, 0.0, 1.0)
            self.computed = True
            return points.tolist(), {"algorithm": "ChaosH"}


class DirectSampling(ContinuousMetaheuristic):

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

    @property
    def search_space(self) -> Direct:
        return self._search_space

    @search_space.setter
    def search_space(self, value: Direct):
        if value:
            if isinstance(value, Direct):
                self.computed = False
                self._search_space = value
                self.center = (value.lower + value.upper) / 2
            else:
                raise InitializationError(
                    f"Search space must be a Hyperpshere, Section, Hypercube, got {type(value)}."
                )
        else:
            raise InitializationError(
                f"Search space must be a Hyperpshere, Section, Hypercube, got {type(value)}."
            )

    def reset(self):
        self.computed = False

    def forward(
        self,
        X: list,
        Y: np.ndarray,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
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
            return [], {"algorithm": "DIRECT"}
        else:
            if self.search_space.level == 0:
                n_points = 2 * len(self.search_space.set_i) + 1
                start = 1
            else:
                n_points = 2 * len(self.search_space.set_i)
                start = 0
            section_length = self.search_space.width / 3

            points = np.tile(self.center, (n_points, 1))

            for i, p1, p2 in zip(
                self.search_space.set_i,
                points[start:-1:2],
                points[start + 1 :: 2],
            ):
                p1[i] -= section_length
                p2[i] += section_length
            self.computed = True
            return points.tolist(), {"algorithm": "DIRECT"}


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
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
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
            return [], {"algorithm": "Base"}
        else:
            self.computed = True
            if self.insertion:
                res = np.full(len(self.base), -1, dtype=int)
                res[: self.fixed_idx] = self.base[: self.fixed_idx]
                return [res.tolist()], {"algorithm": "Base"}
            else:
                return [self.base.tolist()], {"algorithm": "Base"}
