# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:37:15+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import numpy as np

import logging

logger = logging.getLogger("zellij.chaos_map")


class Chaos_map(object):

    """Chaos_map

    :code:`Chaos_map` is in abstract class describing what a chaos map is.

    Attributes
    ----------
    vectors : int
        Size of the map (rows).
    params : int
        Number of parameters (columns).
    map : np.array
        Chaos map of shape (vectors, params).

    See Also
    --------
    Chaotic_optimization : Chaos map is used here.
    """

    def __init__(self, vectors, params):

        self.vectors = vectors
        self.params = params
        self.map = np.zeros([self.vectors, self.params])

    def __add__(self, map):
        assert (
            self.params == map.params
        ), f"Error, maps must have equals `params`, got {self.params} and {map.params}"

        new_map = Chaos_map(self.vectors + map.vectors, self.params)

        new_map.map = np.append(self.map, map.map)
        np.random.shuffle(new_map.map)

        return new_map


class Henon(Chaos_map):
    """Henon chaotic map

    .. math::

        \\smash{
        \\begin{cases}
        \\begin{cases}
        x_{n+1} = 1 - a x_n^2 + y_n\\\\
        y_{n+1} = b x_n.
        \\end{cases}\\\\
        map = \\frac{y-min(y)}{max(y)-min(y)}
        \\end{cases}}

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

    Attributes
    ----------
    map : numpy.ndarray
        Chaos map of size (vectors,param)

    """

    def __init__(self, vectors, params, a=1.4020560, b=0.305620406):

        super().__init__(vectors, params)

        self.a = a
        self.b = b

        # Initialization
        y = np.zeros([self.vectors, self.params])
        x = np.random.random(self.params)

        for i in range(1, self.vectors):

            # y_{k+1} = x_{k}
            y[i, :] = b * x

            # x_{k+1} = a.(1-x_{k}^2) + b.y_{k}
            x = 1 - a * x**2 + y[i - 1, :]

        # Min_{params}(y_{params,vectors})
        alpha = np.amin(y, axis=0)

        # Max_{params}(y_{params,vectors})
        beta = np.amax(y, axis=0)

        self.map = (y - alpha) / (beta - alpha)


class Kent(Chaos_map):
    """Kent chaotic map

    .. math::

        \\smash{
        \\begin{cases}
        x_{n+1} =
        \\begin{cases}
        \\frac{x_n}{\\beta} \\quad 0 < x_{n}  \\leq \\beta \\\\
        \\frac{1-x_n}{1-\\beta} \\quad \\beta < x_{n} \\leq 1
        \\end{cases}\\\\
        map=x
        \\end{cases}}

    Parameters
    ----------
    vectors : int
        Map size
    params : int
        Number of dimensions
    beta : float, default=0.8
        Kent map parameter. Has an influence on the chaotic, intermittent, convergence, or periodicity behaviors.

    Attributes
    ----------
    map : numpy.ndarray
        Chaos map of size (vectors,param)

    """

    def __init__(self, vectors, params, beta=0.8):
        super().__init__(vectors, params)

        self.beta = beta

        self.map[0, :] = np.random.random(params)

        for i in range(1, vectors):
            self.map[i, :] = np.where(
                self.map[i - 1, :] < beta,
                self.map[i - 1, :] / beta,
                (1 - self.map[i - 1, :]) / (1 - beta),
            )


class Logistic(Chaos_map):
    """Logistic chaotic map

    .. math::

        \\smash{
        \\begin{cases}
        x_{n+1} = \\mu x_n(1-x_n)\\\\
        map=x
        \\end{cases}}

    Parameters
    ----------
    vectors : int
        Map size
    params : int
        Number of dimensions
    mu : float, default=3.57
        Logistic map parameter. Has an influence on the chaotic, intermittent, convergence, or periodicity behaviors.

    Attributes
    ----------
    map : numpy.ndarray
        Chaos map of size (vectors,param)

    """

    def __init__(self, vectors, params, mu=3.57):

        super().__init__(vectors, params)

        self.mu = mu

        self.map[0, :] = np.random.random(params)

        for i in range(1, vectors):
            self.map[i, :] = mu * self.map[i - 1, :] * (1 - self.map[i - 1, :])


class Tent(Chaos_map):
    """Tent chaotic map

    .. math::

        \\smash{
        \\begin{cases}
        x_{n+1} =
        \\begin{cases}
        \\mu x_n     \\quad x_n < \\frac{1}{2} \\\\
        \\mu (1-x_n) \\quad \\frac{1}{2} \\leq x_n
        \\end{cases}\\\\
        map = x
        \\end{cases}}

    Parameters
    ----------
    vectors : int
        Map size
    params : int
        Number of dimensions
    Tent : float, default=1.9999999999
        Logistic map parameter. Has an influence on the chaotic, intermittent, convergence, or periodicity behaviors.

    Attributes
    ----------
    map : numpy.ndarray
        Chaos map of size (vectors,param)

    """

    def __init__(self, vectors, params, mu=2 - 1e-10):

        super().__init__(vectors, params)

        self.mu = mu

        self.map[0, :] = np.random.random(params)

        for i in range(1, vectors):
            self.map[i, :] = np.where(
                self.map[i - 1, :] < 0.5,
                self.mu * self.map[i - 1, :],
                self.mu * (1 - self.map[i - 1, :]),
            )


class Random(Chaos_map):
    def __init__(self, vectors, params):
        super().__init__(vectors, params)

        self.map = np.random.random((vectors, params))
