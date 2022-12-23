# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-06T12:07:38+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-11-08T14:26:53+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.addons import VarConverter, Converter
import numpy as np
import logging

logger = logging.getLogger("zellij.converters")
logger.setLevel(logging.INFO)

####################################
# DO NOTHING
####################################


class DoNothing(VarConverter):
    """DoNothing

    :ref:`varadd` used when a :ref:`var` must not be converted.
    It does nothing, excep returning a non converted value.

    """

    def convert(self, value, *args, **kwargs):
        return value

    def reverse(self, value, *args, **kwargs):
        return value


####################################
# TO CONTINUOUS
####################################


class ArrayMinmax(VarConverter):
    """ArrayMinmax

    :ref:`varadd` used when elements of the array must be converted to
    continous.

    Parameters
    ----------
    variable : ArrayVar
        Targeted ArrayVar.

    Attributes
    ----------
    variable : ArrayVar
        Targeted ArrayVar.

    """

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
    """BlockMinmax

    :ref:`varadd` used when elements of the Block must be converted to
    continous.

    Parameters
    ----------
    variable : Block
        Targeted Block.

    Attributes
    ----------
    variable : Block
        Targeted Block.

    """

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
    """DynamicBlockMinmax

    :ref:`varadd` used when elements of the DynamicBlockMinmax must be
    converted to continous.

    Parameters
    ----------
    variable : DynamicBlockMinmax
        Targeted DynamicBlockMinmax.

    Attributes
    ----------
    variable : DynamicBlockMinmax
        Targeted DynamicBlockMinmax.

    """

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
    """FloatMinmax

    Convert the value of a FloatVar, using
    :math:`\\frac{x-lower}{upper-lower}=y`
    .Reverse: :math:`y(upper-lower)+lower=x`

    """

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
    """IntMinmax

    Convert the value of an IntVar, using :math:`\\frac{x-lower}{upper-lower}=y`
    .Reverse: :math:`y(upper-lower)+lower=x`

    """

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
    """CatMinmax

    Convert the value of a CatVar, using the index of the value in the list
    of the features of CatVar. :math:`\\frac{index}{len(features)}=y`
    .Reverse: :math:`features[floor(y*(len(features)-1))]=x`.

    """

    def convert(self, value):

        return self.target.features.index(value) / len(self.target.features)

    def reverse(self, value):
        idx = int(value * len(self.target.features))
        if idx == len(self.target.features):
            idx -= 1

        return self.target.features[idx]


class ConstantMinmax(VarConverter):
    """ConstantMinmax

    Convert the value of a Constant. :math:`y=1.0`
    .Reverse: :math:`x=value`.

    """

    def convert(self, value):
        return 1.0

    def reverse(self, value):
        return self.target.value


####################################
# TO DISCRETE
####################################


class ArrayBinning(VarConverter):
    """ArrayBinning

    :ref:`varadd` used when elements of the array must be converted to
    discrete. When binning some information can be lost.

    Parameters
    ----------
    variable : ArrayVar
        Targeted ArrayVar.

    Attributes
    ----------
    variable : ArrayVar
        Targeted ArrayVar.

    """

    def __init__(self, variable=None):
        super(ArrayBinning, self).__init__(variable)
        if variable:
            assert all(
                hasattr(v, "to_discrete") for v in self.target.values
            ), logger.error(
                f"""
                To use `ArrayMinmax`, values in `ArrayVar` must have a
                `to_discrete` method.
                Use `to_discrete` kwarg when defining a variable
                 """
            )

    def convert(self, value):
        res = []
        for variable, v in zip(self.target.values, value):
            res.append(variable.to_discrete.convert(v))

        return res

    def reverse(self, value):
        res = []
        for variable, v in zip(self.target.values, value):
            res.append(variable.to_discrete.reverse(v))

        return res


class BlockBinning(VarConverter):
    """BlockBinning

    :ref:`varadd` used when elements of the Block must be converted to
    discrete.

    Parameters
    ----------
    variable : Block
        Targeted Block.

    Attributes
    ----------
    variable : Block
        Targeted Block.

    """

    def __init__(self, variable=None):
        super(BlockBinning, self).__init__(variable)

        assert hasattr(self.target.value, "to_discrete"), logger.error(
            f"To use `BlockBinning`, value in `Block` must have a `to_discrete` method. Use `to_discrete` kwarg when defining a variable"
        )

    def convert(self, value):

        assert len(value) == self.target.repeat, logger.error(
            f"Length of value must be equal to `Block` length,\
             got {len(value)}(value)=={self.target.repeat}(Block)"
        )

        res = []
        for v in value:
            res.append(self.target.value.to_discrete.convert(v, K))

        return res

    def reverse(self, value):

        assert len(value) == self.target.repeat, logger.error(
            f"Length of value must be equal to `Block` length,\
             got {len(value)}(value)=={self.target.repeat}(Block)"
        )

        res = []
        for v in value:
            res.append(self.target.value.to_discrete.reverse(v, K))

        return res


class DynamicBlockBinning(VarConverter):
    """DynamicBlockMinmax

    :ref:`varadd` used when elements of the DynamicBlockMinmax must be
    converted to discrete.

    Parameters
    ----------
    variable : DynamicBlockMinmax
        Targeted DynamicBlockMinmax.

    Attributes
    ----------
    variable : DynamicBlockMinmax
        Targeted DynamicBlockMinmax.

    """

    def __init__(self, variable=None):
        super(DynamicBlockBinning, self).__init__(variable)

        assert hasattr(self.target.value, "to_discrete"), logger.error(
            f"To use `DynamicBlockBinning`, value in `DynamicBlock` must have a `to_discrete` method. Use `to_discrete` kwarg when defining a variable"
        )

    def convert(self, value):

        assert len(value) == self.target.repeat, logger.error(
            f"Length of value must be inferior or equal to `DynamicBlock`\
            length, got {len(value)}(value)<={self.target.repeat}(DynamicBlock)"
        )

        res = []
        for v in value:
            res.append(self.target.value.to_discrete.convert(v, K))

        return res

    def reverse(self, value):

        assert len(value) == self.target.repeat, logger.error(
            f"Length of value must be inferior or equal to `DynamicBlock`\
            length, got {len(value)}(value)<={self.target.repeat}(DynamicBlock)"
        )

        res = []
        for v in value:
            res.append(self.value.target.to_discrete.reverse(v, K))

        return res


class FloatBinning(VarConverter):
    """FloatBinning

    Convert a value from an FloatVar using binning between its
    upper and lower bounds. Reversing a converted value will not return the
    initial value. When binning some information can be lost. here the decimal
    part of the float number.

    """

    def __init__(self, K, variable=None):
        super(FloatBinning, self).__init__(variable)

        assert (
            isinstance(K, int) and K > 1
        ), f"K must be an int >1 for FloatBinning, got {K}"
        self.K = K

    def convert(self, value):
        bins = np.linspace(self.target.low_bound, self.target.up_bound, self.K)
        return np.digitize(value, bins) - 1

    def reverse(self, value):
        bins = np.linspace(self.target.low_bound, self.target.up_bound, self.K)
        return bins[value]


class IntBinning(VarConverter):
    """IntMinmax

    Convert a value from an IntVar using binning between its
    upper and lower bounds. Reversing a converted value will not return the
    initial value. When binning some information can be lost.

    """

    def __init__(self, K, variable=None):
        super(IntBinning, self).__init__(variable)

        assert (
            isinstance(K, int) and K > 1
        ), f"K must be an int >1 for IntBinning, got {K}"
        self.K = K

    def convert(self, value):
        bins = np.linspace(self.target.low_bound, self.target.up_bound, self.K)
        return np.digitize(value, bins) - 1

    def reverse(self, value):
        bins = np.linspace(self.target.low_bound, self.target.up_bound, self.K)
        return bins[value]


class CatBinning(VarConverter):
    """CatMinmax

    Convert the value of a CatVar to its corresponding index in the
    features list.

    """

    def convert(self, value):
        return self.target.features.index(value)

    def reverse(self, value):
        return self.target.features[value]


class ConstantBinning(VarConverter):
    """ConstantMinmax

    Convert the value of a Constant. :math:`y=1`
    Reverse: :math:`x=value`.

    """

    def convert(self, value):
        return 1

    def reverse(self, value):
        return self.target.value


####################################
# SEARCH SPACE CONVERTER
####################################


class Continuous(Converter):
    """Continuous

    Convert :ref:`var` of a :ref:`sp` to continuous.
    to_continuous addon must be implemented for each :ref:`var`.

    Parameters
    ----------
    search_space : :ref:`sp`
        Targeted :ref:`sp`.

    Attributes
    ----------
    target : :ref:`sp`
        Targeted :ref:`sp`.

    """

    def __init__(self, search_space=None):
        super(Continuous, self).__init__(search_space)
        if search_space:
            assert hasattr(self.target.values, "to_continuous"), logger.error(
                f"To use `to_continuous`, values in Searchspace must have a `to_continuous` method. Use `to_continuous` kwarg when defining a variable"
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
            If the search space is a subspace and if True,
            uses the original values to convert, else uses its own bounds.
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
            If the search space is a subspace and if True,
            uses the original values to convert, else uses its own bounds.
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
    """Discrete

    Convert :ref:`var` of a :ref:`sp` to discrete.
    to_discrete addon must be implemented for each :ref:`var`.

    Parameters
    ----------
    search_space : :ref:`sp`
        Targeted :ref:`sp`.

    Attributes
    ----------
    target : :ref:`sp`
        Targeted :ref:`sp`.

    """

    def __init__(self, search_space=None, K=10):
        super(Discrete, self).__init__(search_space)
        if search_space:
            assert hasattr(self.target.values, "to_discrete"), logger.error(
                f"To use `Discrete`, values in Searchspace must have a `to_discrete` method. Use `to_discrete` kwarg when defining a variable"
            )

        self.K = K

    # Convert a point to continuous
    def convert(self, points, sub_values=False):
        """convert(self, points, sub_values=False)

        Convert given points from mixed to discrete

        Parameters
        ----------

        points : {list[list[{int, float, str}, {int, float, str}...], ...],\list[list[float, float...], ...]}
            List of points to convert
        sub_values : boolean, default=True
            If the search space is a subspace and if True,
            uses the original values to convert, else uses its own bounds.
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

        # Mixed to discrete
        for point in points:
            res.append(val.to_discrete.convert(point))

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
            If the search space is a subspace and if True,
            uses the original values to convert, else uses its own bounds.
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

        # Mixed to discrete
        for point in points:
            res.append(val.to_discrete.reverse(point))

        return res
