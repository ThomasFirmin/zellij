from zellij.core.addons import Converter
import numpy as np
import logging

logger = logging.getLogger("zellij.neighborhoods")


class ArrayConverter(Converter):
    def convert(self, value):
        pass

    def reverse(self, value):
        pass


class BlockConverter(Converter):
    def convert(self, value):
        pass

    def reverse(self, value):
        pass


class DBlockConverter(Converter):
    def convert(self, value):
        pass

    def reverse(self, value):
        pass


class FloatMinmax(Converter):
    def convert(self, value):
        pass

    def reverse(self, value):
        pass


class IntMinmax(Converter):
    def convert(self, value):
        pass

    def reverse(self, value):
        pass


class CatMinmax(Converter):
    def convert(self, value):
        pass

    def reverse(self, value):
        pass


class FloatBinning(Converter):
    def convert(self, value):
        pass

    def reverse(self, value):
        pass


class IntBinning(Converter):
    def convert(self, value):
        pass

    def reverse(self, value):
        pass


class CatBinning(Converter):
    def convert(self, value):
        pass

    def reverse(self, value):
        pass


class Minmax(Converter):
    def __init__(self, search_space):
        super(Minmax, self).__init__(search_space)

    # Convert a point to continuous
    def convert(self, points, sub_values=False):
        """convert(self, points, sub_values=False)

        Convert given points from mixed to continuous

        Parameters
        ----------

        points : {list[list[{int, float, str}, {int, float, str}...], ...], list[list[float, float...], ...]}
            List of points to convert
        sub_values : boolean, default=True
            If the search space is a subspace, uses the original values to convert if True, else uses its own bounds.
            See :ref:`sp`

        Returns
        -------

        points : {list[list[{int, float, str}, {int, float, str}...], ...], list[list[float, float...], ...]}
            List of converted points. Points are list of float if converted to continuous.

        """

        # Use bounds from the original space if this object is a subspace.
        if sub_values and self.search_space.sub_values != None:
            val = self.search_space.sub_values

        # Use initial bounds to convert
        else:
            val = self.search_space.values

        res = []

        # Mixed to continuous
        for point in points:
            converted = []
            for att in range(self.search_space.n_variables):

                if (
                    self.search_space.types[att] == "R"
                    or self.search_space.types[att] == "D"
                ):

                    converted.append(
                        (point[att] - val[att][0]) / (val[att][1] - val[att][0])
                    )

                elif self.search_space.types[att] == "C":

                    idx = self.search_space.values[att].index(point[att])
                    n_values = len(val[att])

                    converted.append(idx / n_values)

                elif self.search_space.types[att] == "K":
                    converted.append(1)

            res.append(converted[:])

        return res

    # Convert a continuous point to a mixed point
    def reverse(self, points, sub_values=False):
        """reverse(self, points, sub_values=False)

        Convert given points from continuous to mixed

        Parameters
        ----------

        points : {list[list[{int, float, str}, {int, float, str}...], ...], list[list[float, float...], ...]}
            List of points to convert
        sub_values : boolean, default=True
            If the search space is a subspace, uses the original values to convert if True, else uses its own bounds.
            See :ref:`sp`

        Returns
        -------

        points : {list[list[{int, float, str}, {int, float, str}...], ...], list[list[float, float...], ...]}
            List of converted points. Points are list of float if converted to continuous.

        """

        # Use bounds from the original space if this object is a subspace.
        if sub_values and self.search_space.sub_values != None:
            val = self.search_space.sub_values

        # Use initial bounds to convert
        else:
            val = self.search_space.values

        res = []

        # Continuous to mixed
        for point in points:
            converted = []
            for att in range(self.search_space.n_variables):

                if self.search_space.types[att] == "R":

                    converted.append(
                        point[att] * (val[att][1] - val[att][0]) + val[att][0]
                    )

                elif self.search_space.types[att] == "D":

                    converted.append(
                        int(
                            point[att] * (val[att][1] - val[att][0])
                            + val[att][0]
                        )
                    )

                elif self.search_space.types[att] == "C":

                    n_values = len(val[att])
                    converted.append(val[att][int(point[att] * n_values)])

                elif self.search_space.types[att] == "K":
                    converted.append(self.search_space.values[att])

            res.append(converted[:])

        return res


class Binning(Converter):
    def __init__(self, search_space, K):
        super(Binning, self).__init__(search_space)

        # if a list is given
        if isinstance(K, list):
            assert len(K) > 0, logger.error(
                "K must be a list of length > 0, a float or an int,\
                float: type='R', int: type='D', -1: type='C' or 'K'"
            )

            for n, t in zip(K, self.search_space.types):
                if t == "R":
                    assert isinstance(n, int) and n > 0, logger.error(
                        f"For type 'R', K must be an int > 0, got {n}"
                    )

                elif t == "D":
                    assert isinstance(n, int) and (
                        n > 0 or n == -1
                    ), logger.error(
                        f"For type 'D', K must be an int > 0 or =-1, got {n}"
                    )

                else:
                    assert n == -1, logger.error(
                        f"For type 'C' or 'K', K must be equal to -1, got {n}"
                    )

        else:
            assert isinstance(K, int) or isinstance(K, float), logger.error(
                "K must be a list of length > 0, a float or an int,\
                float: type='R', int: type='D', -1: type='C' or 'K'"
            )
            K = [K] * self.search_space.n_variables

            # Update K, -1 for type !="R"
            for t, idx in enumerate(self.search_space.types):
                if t == "D" or t == "C" or t == "K":
                    K[idx] = -1

        self.K = K

        # Compute all bins
        self.bins = [None] * self.search_space.n_variables
        for v, k, idx in zip(
            self.search_space.values, K, range(self.search_space.n_variables)
        ):
            if k != -1:
                self.bins[idx] = np.linspace(v[0], v[1], k)

    # Convert a point to continuous
    def convert(self, points, sub_values=False):
        """convert(self, points, sub_values=False)

        Convert given points from mixed to discrete

        Parameters
        ----------

        points : {list[list[{int, float, str}, {int, float, str}...], ...],\list[list[float, float...], ...]}
            List of points to convert
        sub_values : boolean, default=True
            If the search space is a subspace, uses the original values to convert if True, else uses its own bounds.
            See :ref:`sp`

        Returns
        -------

        points : {list[list[{int, float, str}, {int, float, str}...], ...], list[list[float, float...], ...]}
            List of converted points. Points are list of float if converted to continuous.

        """

        # Use bounds from the original space if this object is a subspace.
        if sub_values and self.search_space.sub_values != None:
            val = self.search_space.sub_values

        # Use initial bounds to convert
        else:
            val = self.search_space.values

        res = []

        # Mixed to discrete
        for point in points:
            converted = []
            for bin, idx in enumerate(self.bins):

                if bin:
                    converted.append(np.digitize(point[idx], bin))

                elif self.search_space.types[idx] == "K":
                    converted.append(point[idx])

            res.append(converted[:])

        return res

    # Convert a continuous point to a mixed point
    def reverse(self, points, sub_values=False):
        """reverse(self, points, sub_values=False)

        Convert given points from continuous to mixed

        Parameters
        ----------

        points : {list[list[{int, float, str}, {int, float, str}...], ...], list[list[float, float...], ...]}
            List of points to convert
        sub_values : boolean, default=True
            If the search space is a subspace, uses the original values to convert if True, else uses its own bounds.
            See :ref:`sp`

        Returns
        -------

        points : {list[list[{int, float, str}, {int, float, str}...], ...], list[list[float, float...], ...]}
            List of converted points. Points are list of float if converted to continuous.

        """

        # Use bounds from the original space if this object is a subspace.
        if sub_values and self.search_space.sub_values != None:
            val = self.search_space.sub_values

        # Use initial bounds to convert
        else:
            val = self.search_space.values

        res = []

        # Continuous to mixed
        for point in points:
            converted = []
            for att in range(self.search_space.n_variables):

                if self.search_space.types[att] == "R":

                    converted.append(
                        point[att] * (val[att][1] - val[att][0]) + val[att][0]
                    )

                elif self.search_space.types[att] == "D":

                    converted.append(
                        int(
                            point[att] * (val[att][1] - val[att][0])
                            + val[att][0]
                        )
                    )

                elif self.search_space.types[att] == "C":

                    n_values = len(val[att])
                    converted.append(val[att][int(point[att] * n_values)])

                elif self.search_space.types[att] == "K":
                    converted.append(self.search_space.values[att])

            res.append(converted[:])

        return res
