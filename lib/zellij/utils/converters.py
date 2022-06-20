# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-06T12:07:38+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-06-13T15:58:44+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


from zellij.core.addons import VarConverter, Converter
import numpy as np
import logging

logger = logging.getLogger("zellij.neighborhoods")
logger.setLevel(logging.INFO)

####################################
# DO NOTHING
####################################


class DoNothing(VarConverter):
    def convert(self, value):
        return value

    def reverse(self, value):
        return value


class ArrayNothing(DoNothing):
    pass


class BlockNothing(DoNothing):
    pass


class DynamicBlockNothing(DoNothing):
    pass


class FloatNothing(DoNothing):
    pass


class IntNothing(DoNothing):
    pass


class CatNothing(DoNothing):
    pass


####################################
# TO CONTINUOUS
####################################


class ArrayMinmax(VarConverter):
    def __init__(self, variable=None):
        super(ArrayMinmax, self).__init__(variable)
        if variable:
            assert all(
                hasattr(v, "to_continuous") for v in self.target.values
            ), logger.error(
                f"To use `ArrayMinmax`, values in `ArrayVar` must have a `to_continuous` method. Use `to_continuous` kwarg when defining a variable"
            )

    def convert(self, value):
        res = []

        for variable, v in zip(self.target.values, value):
            res.append(variable.to_continuous.convert(v))

        return res

    def reverse(self, value):
        res = []

        for variable, v in zip(self.target.values, value):
            res.append(variable.to_continuous.reverse(v))

        return res


class BlockMinmax(VarConverter):
    def __init__(self, variable=None):
        super(BlockMinmax, self).__init__(variable)

        assert hasattr(self.target.value, "to_continuous"), logger.error(
            f"To use `BlockMinmax`, value in `Block` must have a `to_continuous` method. Use `to_continuous` kwarg when defining a variable"
        )

    def convert(self, value):

        assert len(value) == self.target.repeat, logger.error(
            f"Length of value must be equal to `Block` length,\
             got {len(value)}(value)=={self.target.repeat}(Block)"
        )

        res = []
        for v in value:
            res.append(self.target.value.to_continuous.convert(v))

        return res

    def reverse(self, value):

        assert len(value) == self.target.repeat, logger.error(
            f"Length of value must be equal to `Block` length,\
             got {len(value)}(value)=={self.target.repeat}(Block)"
        )

        res = []
        for v in value:
            res.append(self.target.value.to_continuous.reverse(v))

        return res


class DynamicBlockMinmax(VarConverter):
    def __init__(self, variable=None):
        super(DynamicBlockMinmax, self).__init__(variable)

        assert hasattr(self.target.value, "to_continuous"), logger.error(
            f"To use `DynamicBlockMinmax`, value in `DynamicBlock` must have a `to_continuous` method. Use `to_continuous` kwarg when defining a variable"
        )

    def convert(self, value):

        assert len(value) == self.target.repeat, logger.error(
            f"Length of value must be inferior or equal to `DynamicBlock`\
            length, got {len(value)}(value)<={self.target.repeat}(DynamicBlock)"
        )

        res = []
        for v in value:
            res.append(self.target.value.to_continuous.convert(v))

        return res

    def reverse(self, value):

        assert len(value) == self.target.repeat, logger.error(
            f"Length of value must be inferior or equal to `DynamicBlock`\
            length, got {len(value)}(value)<={self.target.repeat}(DynamicBlock)"
        )

        res = []
        for v in value:
            res.append(self.target.value.to_continuous.reverse(v))

        return res


class FloatMinmax(VarConverter):
    def convert(self, value):
        return (value - self.target.low_bound) / (
            self.target.up_bound - self.target.low_bound
        )

    def reverse(self, value):
        return (
            value * (self.target.up_bound - self.target.low_bound)
            + self.target.low_bound
        )


class IntMinmax(VarConverter):
    def convert(self, value):
        return (value - self.target.low_bound) / (
            self.target.up_bound - self.target.low_bound
        )

    def reverse(self, value):
        return int(
            value * (self.target.up_bound - self.target.low_bound)
            + self.target.low_bound
        )


class CatMinmax(VarConverter):
    def convert(self, value):

        return self.target.features.index(value) / len(self.target.features)

    def reverse(self, value):
        return self.target.features[int(value * len(self.target.features))]


class ConstantMinmax(VarConverter):
    def convert(self, value):
        return 1.0

    def reverse(self, value):
        return self.target.value


####################################
# TO DISCRETE
####################################


class ArrayBinning(VarConverter):
    def __init__(self, variable=None):
        super(ArrayBinning, self).__init__(variable)

        assert all(
            hasattr(v, "to_discrete") for v in self.target.values
        ), logger.error(
            f"To use `ArrayBinning`, values in `ArrayVar` must have a `to_discrete` method. Use `to_discrete` kwarg when defining a variable"
        )

    def convert(self, value, K):

        if isinstance(K, list):
            assert len(K) == len(value), logger.error(
                f"length of `K`(number of bins) must be equal to\
                `ArrayVar` length, got {K}=={len(value)}"
            )
        else:
            K = [K] * len(value)

        res = []

        for variable, v, k in zip(self.target.values, value, K):
            res.append(variable.to_discrete.convert(v, k))

        return res

    def reverse(self, value, K):
        res = []

        if isinstance(K, list):
            assert len(K) == len(value), logger.error(
                f"length of `K`(number of bins) must be equal to\
                `ArrayVar` length, got {K}=={len(value)}"
            )
        else:
            K = [K] * len(value)

        for variable, v, k in zip(self.target.values, value, K):
            res.append(variable.to_discrete.reverse(v, k))

        return res


class BlockBinning(VarConverter):
    def __init__(self, variable=None):
        super(BlockBinning, self).__init__(variable)

        assert hasattr(self.target.value, "to_discrete"), logger.error(
            f"To use `BlockBinning`, value in `Block` must have a `to_discrete` method. Use `to_discrete` kwarg when defining a variable"
        )

    def convert(self, value, K):

        assert len(value) == self.target.repeat, logger.error(
            f"Length of value must be equal to `Block` length,\
             got {len(value)}(value)=={self.target.repeat}(Block)"
        )

        res = []
        for v in value:
            res.append(self.target.value.to_discrete.convert(v, K))

        return res

    def reverse(self, value, K):

        assert len(value) == self.target.repeat, logger.error(
            f"Length of value must be equal to `Block` length,\
             got {len(value)}(value)=={self.target.repeat}(Block)"
        )

        res = []
        for v in value:
            res.append(self.target.value.to_discrete.reverse(v, K))

        return res


class DynamicBlockBinning(VarConverter):
    def __init__(self, variable=None):
        super(DynamicBlockBinning, self).__init__(variable)

        assert hasattr(self.target.value, "to_discrete"), logger.error(
            f"To use `DynamicBlockBinning`, value in `DynamicBlock` must have a `to_discrete` method. Use `to_discrete` kwarg when defining a variable"
        )

    def convert(self, value, K):

        assert len(value) == self.target.repeat, logger.error(
            f"Length of value must be inferior or equal to `DynamicBlock`\
            length, got {len(value)}(value)<={self.target.repeat}(DynamicBlock)"
        )

        res = []
        for v in value:
            res.append(self.target.value.to_discrete.convert(v, K))

        return res

    def reverse(self, value, K):

        assert len(value) == self.target.repeat, logger.error(
            f"Length of value must be inferior or equal to `DynamicBlock`\
            length, got {len(value)}(value)<={self.target.repeat}(DynamicBlock)"
        )

        res = []
        for v in value:
            res.append(self.value.target.to_discrete.reverse(v, K))

        return res


class FloatBinning(VarConverter):
    def convert(self, value, K):
        bins = np.linspace(self.target.low_bound, self.target.up_bound, K)
        return np.digitize(value, bins)

    def reverse(self, value, K):
        bins = np.linspace(self.target.low_bound, self.target.up_bound, K)

        return bins[value]


class IntBinning(VarConverter):
    def convert(self, value, K):
        return value

    def reverse(self, value, K):
        return value


class CatBinning(VarConverter):
    def convert(self, value, K):
        return self.target.features.index(value)

    def reverse(self, value, K):
        return self.target.features[value]


class ConstantBinning(VarConverter):
    def convert(self, value, K):
        return 1

    def reverse(self, value, K):
        return self.target.value


####################################
# SEARCH SPACE CONVERTER
####################################


class Continuous(Converter):
    def __init__(self, search_space=None):
        super(Continuous, self).__init__(search_space)
        if search_space:
            assert hasattr(self.target.values, "to_continuous"), logger.error(
                f"To use `Intervals`, values in Searchspace must have a `to_continuous` method. Use `neighbor` kwarg when defining a variable"
            )

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
        if sub_values and self.target._god.values != None:
            val = self.target._god.values

        # Use initial bounds to convert
        else:
            val = self.target.values

        res = []

        # Mixed to continuous
        if isinstance(points[0], (list, np.ndarray)):
            res = []
            for point in points:
                res.append(val.to_continuous.convert(point))

            return res
        else:
            return val.to_continuous.convert(points)

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
        if sub_values and self.target._god.values != None:
            val = self.target._god.values

        # Use initial bounds to convert
        else:
            val = self.target.values

        # Mixed to continuous
        if isinstance(points[0], (list, np.ndarray)):
            res = []
            for point in points:
                res.append(val.to_continuous.reverse(point))

            return res
        else:
            return val.to_continuous.reverse(points)


class Discrete(Converter):
    def __init__(self, search_space, K):
        super(Discrete, self).__init__(search_space)
        assert hasattr(self.target.values, "to_continuous"), logger.error(
            f"To use `Intervals`, values in Searchspace must have a `to_discrete` method. Use `neighbor` kwarg when defining a variable"
        )

        # if a list is given
        if isinstance(K, list):
            assert len(K) == len(search_space.size), logger.error(
                f"length of `K`(number of bins) must be equal to\
                `ArrayVar` length, got {K}=={search_space.size}"
            )

            self.K = K
        else:
            self.K = [K] * search_space.size

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
        if sub_values and self.search_space._god.values != None:
            val = self.search_space._god.values

        # Use initial bounds to convert
        else:
            val = self.search_space.values

        res = []

        # Mixed to discrete
        for point in points:
            res.append(val.to_discrete.convert(point, self.K))

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
        if sub_values and self.search_space._god.values != None:
            val = self.search_space._god.values

        # Use initial bounds to convert
        else:
            val = self.search_space.values

        res = []

        # Mixed to discrete
        for point in points:
            res.append(val.to_discrete.reverse(point, self.K))

        return res
