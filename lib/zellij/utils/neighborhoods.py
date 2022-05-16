from zellij.core.addons import Neighborhood
import numpy as np
import logging

logger = logging.getLogger("zellij.neighborhoods")


class Intervals(Neighborhood):
    def __init__(self, search_space, neighborhood):
        super(Intervals, self).__init__(search_space, neighborhood)

    @Neighborhood.neighborhood.setter
    def neighborhood(self, neighborhood):
        assert (
            isinstance(neighborhood, list) and len(neighborhood) > 0
        ), logger.error(
            "neighborhood must be of the form [float|int|-1, ...],\
            float: type='R', int: type='D', -1: type='C' or 'K'"
        )

        for n, t in zip(neighborhood, self.search_space.types):
            if t == "R":
                assert isinstance(n, float) or isinstance(n, int), logger.error(
                    f"neighborhood of type 'R'\
                    must be a float or an int, got {n}"
                )

            elif t == "D":
                assert isinstance(n, int), logger.error(
                    f"neighborhood of type 'R'\
                     must be an int, got {n}"
                )

            else:
                assert n == -1, logger.error(
                    f"neighborhood of type 'C' or 'K'\
                     must be equal to -1, got {n}"
                )

        self._neighborhood = neighborhood

    def _get_real_neighbor(self, x, i):

        """_get_real_neighbor(x, i)

        Draw a neighbor of a Real attribute from the search space,\
        using uniform distribution. According to its lower and upper bounds

        Parameters
        ----------

        x : float
            Initial value
        i : int
            Dimension index

        Returns
        -------

        v : float
            Random neighbor of x

        """

        upper = np.min(
            [x + self.neighborhood[i], self.search_space.values[i][1]]
        )
        lower = np.max(
            [x - self.neighborhood[i], self.search_space.values[i][0]]
        )
        v = np.random.uniform(lower, upper)

        while v == x:
            v = np.random.uniform(lower, upper)

        return float(v)

    def _get_discrete_neighbor(self, x, i):

        """_get_discrete_neighbor(x, i)

        Draw a neighbor of a Discrete attribute from the search space, using\
        discrete uniform distribution. According to its lower and upper bounds

        Parameters
        ----------

        x : float
            Initial value
        i : int
            Dimension index

        Returns
        -------

        v : int
            Random neighbor of x

        """

        upper = (
            int(
                np.min(
                    [x + self.neighborhood[i], self.search_space.values[i][1]]
                )
            )
            + 1
        )
        lower = int(
            np.max([x - self.neighborhood[i], self.search_space.values[i][0]])
        )
        v = np.random.randint(lower, upper)

        while v == x:
            v = np.random.randint(lower, upper)

        return int(v)

    def _get_categorical_neighbor(self, x, i):

        """_get_categorical_neighbor(self, x, i)

        Draw a neighbor of a Categorical attribute from the search space,\
        using discrete uniform distribution. According to all its possible value

        Parameters
        ----------

        x : float
            Initial value
        i : int
            Dimension index

        Returns
        -------

        v : float
            Random neighbor of x

        """

        idx = np.random.randint(len(self.search_space.values[i]))

        while self.search_space.values[i][idx] == x:
            idx = np.random.randint(len(self.search_space.values[i]))

        v = self.search_space.values[i][idx]

        return v

    def get_neighbor(self, point, size=1):

        """get_neighbor(point, size=1)

        Draw a neighbor of a solution, according to the search space bounds and\
        dimensions types.

        Parameters
        ----------

        point : list[{int, float, str}, {int, float, str}...]
            Initial point.
        size : int, default=1
            Draw <size> neighbors of <point>.

        Returns
        -------

        points : list[list[{int, float, str}]]
            List of neighbors of <point>.

        """

        points = []

        for _ in range(size):
            attribute = self.search_space.random_attribute()
            index = self.search_space.labels.index(attribute)
            neighbor = point[:]

            if self.search_space.types[index] == "R":
                neighbor[index] = self._get_real_neighbor(point[index], index)

            elif self.search_space.types[index] == "D":

                neighbor[index] = self._get_discrete_neighbor(
                    point[index], index
                )

            elif self.search_space.types[index] == "C":

                neighbor[index] = self._get_categorical_neighbor(
                    point[index], index
                )

            points.append(neighbor[:])

        return points
