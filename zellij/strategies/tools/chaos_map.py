# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from abc import ABC, abstractmethod
import contextlib
from typing import Optional

from zellij.core.errors import DimensionalityError

import numpy as np

import logging

logger = logging.getLogger("zellij.chaos_map")


class ChaosMap(ABC):

    """ChaosMap

    :code:`ChaosMap` is in abstract class describing what a chaos map is.
    It is used to sample solutions.

    Attributes
    ----------
    nvectors : int
        Map size (rows).
    params : int
        Number of parameters (columns).
    map : np.array
        Chaos map of shape (vectors, params).
    seed
        Seed of numpy RNG.

    See Also
    --------
    ChaoticOptimization : Chaos map is used here.
    """

    def __init__(self, nvectors: int, params: int, seed=None):
        self.nvectors = nvectors
        self.params = params
        self.seed = seed

        self.map = np.zeros([self.nvectors, self.params])

    @abstractmethod
    def sample(self, seed: Optional[int] = None):
        """sample

        Sample random chaotic map.

        Parameters
        ----------
        seed : int
            Seed for RNG.
        """
        pass

    @contextlib.contextmanager
    def _temp_seed(self, seed: Optional[int]):
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)

    def __add__(self, other: ChaosMap):
        if self.params != other.params:
            raise DimensionalityError(
                f"Error, maps must have equals `params`, got {self.params} and {other.params}"
            )

        new_map = _AddedMap(self.nvectors + other.nvectors, self.params, self, other)

        return new_map


class _AddedMap(ChaosMap):
    """_AddedMap

    Addition of two maps. Maps are shuffled.

    """

    def __init__(
        self, nvectors: int, params: int, map1: ChaosMap, map2: ChaosMap, seed=None
    ):
        super().__init__(nvectors, params, seed)
        self.map[: map1.nvectors, : map1.params] = map1.map
        self.map[map2.nvectors :, map2.params :] = map2.map

    def sample(self, seed: Optional[int]):
        # shuffles maps inplace
        with self._temp_seed(seed if seed else self.seed):
            np.random.shuffle(self.map)


class Henon(ChaosMap):
    """Henon chaotic map

    .. math::
        \\begin{cases}
        \\begin{cases}
        x_{n+1} = 1 - a x_n^2 + y_n\\\\
        y_{n+1} = b x_n.
        \\end{cases}\\\\
        map = \\frac{y-min(y)}{max(y)-min(y)}
        \\end{cases}

    Parameters
    ----------
    nvectors : int
        Map size
    params : int
        Number of dimensions
    a : float, default=1.4020560
        Henon map parameter. Has an influence on the chaotic, intermittent or periodicity behaviors.
    b : float, default=0.305620406
        Henon map parameter. Has an influence on the chaotic, intermittent or periodicity behaviors.

    Attributes
    ----------
    map : numpy.ndarray
        Chaos map of size (vectors,param)

    Examples
    --------
    >>> from zellij.strategies.tools import Henon
    >>> cmap = Henon(5,2)
    >>> print(cmap.map.shape)
    (5, 2)
    >>> print(cmap.map)
    [[0.37113568 0.1903755 ]
    [0.7771203  0.28780393]
    [0.52583348 0.852618  ]
    [1.         0.        ]
    [0.         1.        ]]
    """

    def __init__(
        self,
        nvectors: int,
        params: int,
        a: float = 1.4020560,
        b: float = 0.305620406,
        seed=None,
    ):
        super().__init__(nvectors, params, seed)

        self.a = a
        self.b = b
        self.sample(self.seed)

    def sample(self, seed: Optional[int] = None):
        with self._temp_seed(seed if seed else self.seed):
            # Initialization
            y = np.zeros([self.nvectors, self.params])
            x = np.random.random(self.params)

            for i in range(1, self.nvectors):
                # y_{k+1} = x_{k}
                y[i, :] = self.b * x

                # x_{k+1} = a.(1-x_{k}^2) + b.y_{k}
                x = 1 - self.a * x**2 + y[i - 1, :]

            # Min_{params}(y_{params,vectors})
            alpha = np.amin(y, axis=0)

            # Max_{params}(y_{params,vectors})
            beta = np.amax(y, axis=0)

            self.map = (y - alpha) / (beta - alpha)


class Kent(ChaosMap):
    """Kent chaotic map

    .. math::

        \\begin{cases}
        x_{n+1} =
        \\begin{cases}
        \\frac{x_n}{\\beta} \\quad 0 < x_{n}  \\leq \\beta \\\\
        \\frac{1-x_n}{1-\\beta} \\quad \\beta < x_{n} \\leq 1
        \\end{cases}\\\\
        map=x
        \\end{cases}

    Parameters
    ----------
    nvectors : int
        Map size.
    params : int
        Number of dimensions.
    beta : float, default=0.8
        Kent map parameter. Has an influence on the chaotic, intermittent,
        convergence, or periodicity behaviors.
    seed : int
        Seed for the numpy RNG.

    Attributes
    ----------
    map : numpy.ndarray
        Chaos map of size (vectors,param)

    Examples
    --------
    >>> from zellij.strategies.tools import Kent
    >>> cmap = Kent(5,2)
    >>> print(cmap.map.shape)
    (5, 2)
    >>> print(cmap.map)
    [[0.27544269 0.90451904]
    [0.34430336 0.47740478]
    [0.4303792  0.59675598]
    [0.53797399 0.74594498]
    [0.67246749 0.93243122]]
    
    """

    def __init__(
        self, nvectors: int, params: int, beta: float = 0.8, seed: Optional[int] = None
    ):
        super().__init__(nvectors, params, seed)

        self.beta = beta
        self.sample(self.seed)

    def sample(self, seed: Optional[int] = None):
        with self._temp_seed(seed if seed else self.seed):
            self.map[0, :] = np.random.random(self.params)

            for i in range(1, self.nvectors):
                self.map[i, :] = np.where(
                    self.map[i - 1, :] < self.beta,
                    self.map[i - 1, :] / self.beta,
                    (1 - self.map[i - 1, :]) / (1 - self.beta),
                )


class Logistic(ChaosMap):
    """Logistic chaotic map

    .. math::

        \\begin{cases}
        x_{n+1} = \\mu x_n(1-x_n)\\\\
        map=x
        \\end{cases}

    Parameters
    ----------
    vectors : int
        Map size.
    params : int
        Number of dimensions.
    mu : float, default=3.57
        Logistic map parameter. Has an influence on the chaotic, intermittent,
        convergence, or periodicity behaviors.
    seed : int
        Seed for the numpy RNG.

    Attributes
    ----------
    map : numpy.ndarray
        Chaos map of size (vectors,param)

    Examples
    --------
    >>> from zellij.strategies.tools import Logistic
    >>> cmap = Logistic(5,2)
    >>> print(cmap.map.shape)
    (5, 2)
    >>> print(cmap.map)
    [[0.91074398 0.97705604]
    [0.2902031  0.08003061]
    [0.73536739 0.26284379]
    [0.69472983 0.69171224]
    [0.75712665 0.7612897 ]]
    
    """

    def __init__(
        self, nvectors: int, params: int, mu: float = 3.57, seed: Optional[int] = None
    ):
        super().__init__(nvectors, params, seed)
        self.mu = mu
        self.sample(self.seed)

    def sample(self, seed: Optional[int] = None):
        with self._temp_seed(seed if seed else self.seed):
            self.map[0, :] = np.random.random(self.params)

            for i in range(1, self.nvectors):
                self.map[i, :] = self.mu * self.map[i - 1, :] * (1 - self.map[i - 1, :])


class Tent(ChaosMap):
    """Tent chaotic map

    .. math::

        \\begin{cases}
        x_{n+1} =
        \\begin{cases}
        \\mu x_n     \\quad x_n < \\frac{1}{2} \\\\
        \\mu (1-x_n) \\quad \\frac{1}{2} \\leq x_n
        \\end{cases}\\\\
        map = x
        \\end{cases}

    Parameters
    ----------
    vectors : int
        Map size
    params : int
        Number of dimensions
    mu : float, default=1.9999999999
        Logistic map parameter. Has an influence on the chaotic, intermittent,
        convergence, or periodicity behaviors.
    seed : int
        Seed for the random number generator.

    Attributes
    ----------
    map : numpy.ndarray
        Chaos map of size (vectors,param)

    Examples
    --------
    >>> from zellij.strategies.tools import Logistic
    >>> cmap = Logistic(5,2)
    >>> print(cmap.map.shape)
    (5, 2)
    >>> print(cmap.map)
    [[0.3254483  0.81079411]
    [0.6508966  0.37841178]
    [0.6982068  0.75682356]
    [0.60358639 0.48635288]
    [0.79282721 0.97270577]]
    
    """

    def __init__(
        self,
        nvectors: int,
        params: int,
        mu: float = 1.9999999999,
        seed: Optional[int] = None,
    ):
        super().__init__(nvectors, params, seed)
        self.mu = mu
        self.sample(self.seed)

    def sample(self, seed: Optional[int] = None):
        with self._temp_seed(seed if seed else self.seed):
            self.map[0, :] = np.random.random(self.params)
            for i in range(1, self.nvectors):
                self.map[i, :] = np.where(
                    self.map[i - 1, :] < 0.5,
                    self.mu * self.map[i - 1, :],
                    self.mu * (1 - self.map[i - 1, :]),
                )


class Random(ChaosMap):
    def __init__(self, nvectors: int, params: int, seed: Optional[int] = None):
        super().__init__(nvectors, params, seed)
        self.sample(seed)

    def sample(self, seed: Optional[int] = None):
        with self._temp_seed(seed if seed else self.seed):
            self.map = np.random.random((self.nvectors, self.params))
