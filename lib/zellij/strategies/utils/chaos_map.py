import numpy as np
import matplotlib.pyplot as plt


class Chaos_map(object):

    """Chaos_map Global search

    Chaos_map is used to build the chaotic map according to the give name, the level and dimensions of the Searchspace.
    See Chaotic_optimization for more info. Th chaotic map is matrix of shape: (level, dimension).

    Attributes
    ----------
    level : int
        Chaotic level corresponds to the number of iteration of the chaotic map
    dimension : int
        Dimension of the Searchspace.
    map_kwargs : dict
        Keyword arguments associated to the selected map.
    map : np.array(dtype=np.array(dtype=float))
        Contains the chaotic map of size level*dimension.

    See Also
    --------
    Chaotic_optimization : Chaos map is used here.
    """

    def __init__(self, vectors, params):

        self.vectors = vectors
        self.params = params
        self.map = np.zeros([self.vectors, self.params])

    def __add__(self, map):
        assert self.params == map.params, f"Error, maps must have equals `params`, got {self.params} and {map.params}"

        new_map = Chaos_map(self.vectors + map.vectors, self.params)

        new_map.map = np.append(self.map, map.map)
        np.random.shuffle(new_map.map)

        return new_map


class Henon(Chaos_map):
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
            x = 1 - a * x ** 2 + y[i - 1, :]

        # Min_{params}(y_{params,vectors})
        alpha = np.amin(y, axis=0)

        # Max_{params}(y_{params,vectors})
        beta = np.amax(y, axis=0)

        self.map = (y - alpha) / (beta - alpha)


class Logistic(Chaos_map):
    def __init__(self, vectors, params, mu=3.57):

        super().__init__(vectors, params)

        self.mu = mu

        self.map[0, :] = np.random.random(params)

        for i in range(1, vectors):
            self.map[i, :] = mu * self.map[i - 1, :] * (1 - self.map[i - 1, :])


class Kent(Chaos_map):
    def __init__(self, vectors, params, beta=0.8):
        super().__init__(vectors, params)

        self.beta = beta

        self.map[0, :] = np.random.random(params)

        for i in range(1, vectors):
            self.map[i, :] = np.where(self.map[i - 1, :] < beta, self.map[i - 1, :] / beta, (1 - self.map[i - 1, :]) / (1 - beta))


class Tent(Chaos_map):
    def __init__(self, vectors, params, mu=0.8):

        super().__init__(vectors, params)

        self.mu = mu

        for i in range(1, vectors):
            self.map[i, :] = mu * (1 - 2 * np.absolute((self.map[i - 1, :] - 0.5)))


class Random(Chaos_map):
    def __init__(self, vectors, params):
        super().__init__(vectors, params)

        self.map = np.random.random((vectors, params))


chaos_map_name = {"henon": Henon, "logistic": Logistic, "kent": Kent, "tent": Tent, "random": Random}
