import numpy as np


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

    def __init__(self, name, level, dimension, map_kwargs=None):

        chaos_map_name = {"henon": _henon_map, "logistic": _logistic_map, "kent": _kent_map, "tent": _tent_map, "random": _random_map}

        self.level = level
        self.dimension = dimension
        self.map_kwargs = map_kwargs

        if type(name) == str:

            try:
                self.map = [chaos_map_name[name]]
            except:
                print(f"Invalid map name:{name}. Try:{chaos_map.keys()[1:-1]}")
                exit()
        else:

            for i in name:
                self.map = []

                try:
                    self.map.append(chaos_map_name[i])
                except:
                    print(f"Invalid map name{i}. Try:{chaos_map.keys()[1:-1]}")
                    exit()

        if self.map_kwargs != None:
            self.chaos_map = self.map[0](self.level, self.dimension)
            for m in self.map[1:]:
                res = m(self.level, self.dimension, **self.map_kwargs)
                self.chaos_map = np.append(self.chaos_map, res, axis=0)
        else:
            self.chaos_map = self.map[0](self.level, self.dimension)
            for m in self.map[1:]:
                res = m(self.level, self.dimension)
                self.chaos_map = np.append(self.chaos_map, res, axis=0)

        if type(name) != str:
            np.random.shuffle(self.chaos_map)

        self.inverted_choas_map = 1 - self.chaos_map


def _henon_map(n_vectors, n_param, a=1.4020560, b=0.305620406):

    # Initialization
    x = np.zeros([n_vectors, n_param])
    y = np.zeros([n_vectors, n_param])

    x[0, :] = np.random.random(n_param)

    for i in range(1, n_vectors):

        # x_{k+1} = a.(1-x_{k}^2) + b.y_{k}
        x[i, :] = 1 - a * np.square(x[i - 1, :]) + y[i - 1, :]

        # y_{k+1} = x_{k}
        y[i, :] = b * x[i - 1, :]

    # Min_{n_param}(y_{n_param,n_vectors})
    alpha = np.amin(y, axis=0)

    # Max_{n_param}(y_{n_param,n_vectors})
    beta = np.amax(y, axis=0)

    return (y - alpha) / (beta - alpha)


def _logistic_map(n_vectors, n_param, mu=3.57):
    x = np.zeros([n_vectors, n_param])
    x[0, :] = np.random.random(n_param)

    for i in range(1, n_vectors):
        x[i, :] = mu * x[i - 1, :] * (1 - x[i - 1, :])

    return x


def _kent_map(n_vectors, n_param, beta=0.8):
    x = np.zeros([n_vectors, n_param])
    x[0, :] = np.random.random(n_param)

    for i in range(1, n_vectors):
        x[i, :] = np.where(x[i - 1, :] < beta, x[i - 1, :] / beta, (1 - x[i - 1, :]) / (1 - beta))

    return x


def _tent_map(n_vectors, n_param, mu=0.8):
    x = np.zeros([n_vectors, n_param])

    for i in range(1, n_vectors):
        x[i, :] = mu * (1 - 2 * np.absolute((x[i - 1, :] - 0.5)))

    return x


def _random_map(n_vectors, n_param):
    return np.random.random((n_vectors, n_param))
