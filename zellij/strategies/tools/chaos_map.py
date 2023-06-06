# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-23T15:43:47+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

import contextlib
import numpy as np
from abc import ABC, abstractmethod

import logging

logger = logging.getLogger("zellij.chaos_map")


class Chaos_map(ABC):

    """Chaos_map

    :code:`Chaos_map` is in abstract class describing what a chaos map is.
    It is used to sample solutions.

    Attributes
    ----------
    vectors : int
        Size of the map (rows).
    params : int
        Number of parameters (columns).
    seed : int
        Seed for the random number generator.
    map : np.array
        Chaos map of shape (vectors, params).

    See Also
    --------
    Chaotic_optimization : Chaos map is used here.
    """

    def __init__(self, vectors, params, seed):
        self.vectors = vectors
        self.params = params
        self.seed = seed

        self.map = np.zeros([self.vectors, self.params])

    @abstractmethod
    def sample(self):
        """sample()

        Create the chaotic map

        """
        pass

    @contextlib.contextmanager
    def _temp_seed(self, seed):
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)

    def __add__(self, other):
        assert (
            self.params == other.params
        ), f"Error, maps must have equals `params`, got {self.params} and {other.params}"

        new_map = _AddedMap(
            self.vectors + other.vectors, self.params, self, other, self.seed
        )

        return new_map


class _AddedMap(Chaos_map):
    """_AddedMap

    Addition of two maps. Maps are shuffled.

    """

    def __init__(self, vectors, params, map1, map2, seed):
        self.vectors = vectors
        self.params = params
        self.seed = seed

        self.map = np.append(map1.map, map2.map)

    def sample(self):
        np.random.shuffle(self.map)


class Henon(Chaos_map):
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
    vectors : int
        Map size
    params : int
        Number of dimensions
    a : float, default=1.4020560
        Henon map parameter. Has an influence on the chaotic, intermittent or periodicity behaviors.
    b : float, default=0.305620406
        Henon map parameter. Has an influence on the chaotic, intermittent or periodicity behaviors.
    seed : int
        Seed for the random number generator.

    Attributes
    ----------
    map : numpy.ndarray
        Chaos map of size (vectors,param)

    """

    def __init__(self, vectors, params, a=1.4020560, b=0.305620406, seed=0):
        super().__init__(vectors, params, seed)

        self.a = a
        self.b = b
        self.sample(seed)

    def sample(self, seed):
        with self._temp_seed(seed):
            # Initialization
            y = np.zeros([self.vectors, self.params])
            x = np.random.random(self.params)

            for i in range(1, self.vectors):
                # y_{k+1} = x_{k}
                y[i, :] = self.b * x

                # x_{k+1} = a.(1-x_{k}^2) + b.y_{k}
                x = 1 - self.a * x**2 + y[i - 1, :]

            # Min_{params}(y_{params,vectors})
            alpha = np.amin(y, axis=0)

            # Max_{params}(y_{params,vectors})
            beta = np.amax(y, axis=0)

            self.map = (y - alpha) / (beta - alpha)


class Kent(Chaos_map):
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
    vectors : int
        Map size
    params : int
        Number of dimensions
    beta : float, default=0.8
        Kent map parameter. Has an influence on the chaotic, intermittent,
        convergence, or periodicity behaviors.
    seed : int
        Seed for the random number generator.

    Attributes
    ----------
    map : numpy.ndarray
        Chaos map of size (vectors,param)

    """

    def __init__(self, vectors, params, beta=0.8, seed=0):
        super().__init__(vectors, params, seed)

        self.beta = beta
        self.sample(seed)

    def sample(self, seed):
        with self._temp_seed(seed):
            self.map[0, :] = np.random.random(self.params)

            for i in range(1, self.vectors):
                self.map[i, :] = np.where(
                    self.map[i - 1, :] < self.beta,
                    self.map[i - 1, :] / self.beta,
                    (1 - self.map[i - 1, :]) / (1 - self.beta),
                )


class Logistic(Chaos_map):
    """Logistic chaotic map

    .. math::

        \\begin{cases}
        x_{n+1} = \\mu x_n(1-x_n)\\\\
        map=x
        \\end{cases}

    Parameters
    ----------
    vectors : int
        Map size
    params : int
        Number of dimensions
    mu : float, default=3.57
        Logistic map parameter. Has an influence on the chaotic, intermittent,
        convergence, or periodicity behaviors.
    seed : int
        Seed for the random number generator.

    Attributes
    ----------
    map : numpy.ndarray
        Chaos map of size (vectors,param)

    """

    def __init__(self, vectors, params, mu=3.57, seed=0):
        super().__init__(vectors, params, seed)

        self.mu = mu
        self.sample(seed)

    def sample(self, seed):
        with self._temp_seed(seed):
            self.map[0, :] = np.random.random(self.params)

            for i in range(1, self.vectors):
                self.map[i, :] = self.mu * self.map[i - 1, :] * (1 - self.map[i - 1, :])


class Tent(Chaos_map):
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

    """

    def __init__(self, vectors, params, mu=2 - 1e-10, seed=0):
        super().__init__(vectors, params, seed)

        self.mu = mu

    def sample(self, seed):
        with self._temp_seed(seed):
            self.map[0, :] = np.random.random(self.params)

            for i in range(1, self.vectors):
                self.map[i, :] = np.where(
                    self.map[i - 1, :] < 0.5,
                    self.mu * self.map[i - 1, :],
                    self.mu * (1 - self.map[i - 1, :]),
                )


class Random(Chaos_map):
    def __init__(self, vectors, params, seed=0):
        super().__init__(vectors, params, seed)
        self.sample(seed)

    def sample(self, seed):
        with self._temp_seed(seed):
            self.map = np.random.random((self.vectors, self.params))
