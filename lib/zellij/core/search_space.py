# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-06-19T11:55:08+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


from zellij.core.variables import Variable, ArrayVar, FloatVar, IntVar
from zellij.core.loss_func import LossFunc
from zellij.utils.distances import Distance, Mixed, Euclidean
from zellij.core.addons import SearchSpaceAddon
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import copy
import os
from abc import ABC, abstractmethod


import logging

logger = logging.getLogger("zellij.space")


class Searchspace(ABC):
    """Searchspace

    Searchspace is an essential class for Zellij. Define your search space with this object.

    Attributes
    ----------

    values : ArrayVar
        Determines the bounds of the search space. For `HpoSearchspace` the
        `values` must be an `ArrayVar`. The :ref:`sp` will then manipulate
        this array.  Be carefull, in HPO paradigm, not all algorithms will
        manager `ArrayVar` composed of `Block` or other `ArrayVar`

    loss : LossFunc
        Callable of type `LossFunc`. See :ref:`lf` for more information.
        `loss` will be used by the :ref:`sp` object and by optimization
        algorithms.

    size : list
        Number of dimensions

    Methods
    -------
    random_attribute(self,size=1,replace=True, exclude=None)
        Draw random features from the search space.
        Return the selected `Variable`

    random_point(self,size=1)
        Return random points from the search space

    subspace(self,lo_bounds,up_bounds)
        Build a sub space according to the actual Searchspace using two vectors
        containing lower and upper bounds of the subspace.

    show(self,X,Y)
        Show solutions X associated to their values Y,
        according to the Searchspace

    See Also
    --------
    LossFunc : Parent class for a loss function.

    Examples
    --------

    """

    def __init__(self, values, loss, **kwargs):
        """__init__(values, loss, **kwargs)

        Parameters
        ----------

        values : ArrayVar
            Determines the bounds of the search space. For `HpoSearchspace` the
            `values` must be an `ArrayVar`. The :ref:`sp` will then manipulate
            this array.  Be carefull, in HPO paradigm, not all algorithms will
            manager `ArrayVar` composed of `Block` or other `ArrayVar`

        loss : LossFunc
            Callable of type `LossFunc`. See :ref:`lf` for more information.
            `loss` will be used by the :ref:`sp` object and by optimization
            algorithms.

        kwargs : dict
            Kwargs are the different addons one want to add to a `Variable`.
            Known addons are:
            * to_discrete : Converter
                * Will be called when converting to discrete is needed.
            * to_continuous : Converter
                * Will be called when converting to continuous is needed.
            * neighbor : Neighborhood
                * Will be called when a neighborhood is needed.
            * distance: Distance, default, Mixed
                * Will be called when computing a distance is needed
        """

        assert isinstance(loss, LossFunc), logger.error(
            f"`loss` must be a `LossFunc`, got {loss}"
        )

        ##############
        # PARAMETERS #
        ##############

        self.values = values

        self.loss = loss
        self.kwargs = kwargs

        #############
        # VARIABLES #
        #############

        self.size = len(self.values)

        self._add_addons(**kwargs)

    def _add_addons(self, **kwargs):
        for k in kwargs:
            assert isinstance(kwargs[k], SearchSpaceAddon), logger.error(
                f"Kwargs must be of type `VarAddon`, got {k}:{kwargs[k]}"
            )

            if kwargs[k]:
                setattr(self, k, copy.copy(kwargs[k]))
                addon = getattr(self, k)
                addon.target = self
            else:
                setattr(self, k, kwargs[k])
                addon = getattr(self, k)
                addon.target = self

    # Return 1 or size=n random attribute from the search space, can exclude one attribute
    def random_attribute(self, size=1, replace=True, exclude=None):

        """random_attribute(size=1,replace=True, exclude=None)

        Draw random features from the search space.

        Parameters
        ----------
        size : int, default=1
            Select randomly <size> features.
        replace : boolean, default=True
            Select randomly <size> features with replacement if True.
            See numpy.random.choice
        exclude : Variable or list[Variable] or type or list[type] or int or list[int], default=None
            Exclude one or several `Variable` to be drawn.
            Can also exclude types.For example one can exclude all Constant type.
            Can also exclude variables according to their index is `values`.

        Returns
        -------

        out : numpy.array(dtype=int)
            Array of index, corresponding to the selected Variable in `values`.

        Examples
        --------
        >>> rand_att = sp.random_attribute(3)
        >>> print(rand_att)
        array([FloatVar(float_1, [2;12]), CatVar(cat_1, ['Hello', 87, 2.56]),
         FloatVar(float_1, [2;12])], dtype=object)
        """

        if exclude:
            if isinstance(exclude, int):
                index = exclude
            elif isinstance(exclude, Variable):
                index = exclude.idx
            elif isinstance(exclude, type):
                index = []
                for elem in self.values.values:
                    if isinstance(elem, exclude):
                        index.append(elem.idx)
            elif isinstance(exclude, list):
                if all(isinstance(elem, int) for elem in exclude):
                    index = exclude
                elif all(isinstance(elem, Variable) for elem in exclude):
                    index = []
                    for elem in exclude:
                        index.append(elem.idx)
                elif all(isinstance(elem, type) for elem in exclude):
                    index = []
                    for elem in self.values.values:
                        if isinstance(elem, tuple(exclude)):
                            index.append(elem.idx)

            p = np.full(self.size, 1 / (self.size - len(index)))
            p[index] = 0
        else:
            p = np.full(self.size, 1 / self.size)

        return np.random.choice(
            self.values.values, size=size, replace=replace, p=p
        )

    # Return a random point of the search space
    def random_point(self, size=1):

        """random_point(size=1)

        Return a random point from the search space

        Parameters
        ----------

        size : int, default=1
            Draw <size> points.

        Returns
        -------

        points : list[list[{int, float, str}]]
            List of <point>.

        Examples
        --------

        >>> rand_pts = sp.random_point(3)
        >>> print(f"Random points: {rand_pts}")
        Random points: [[-3.830114043118622, 9, 'sigmoid'],
        ...             [3.065902630698311, 3, 'sigmoid'],
        ...             [-0.6839762230289024, 10, 'relu']]

        """

        return self.values.random(size)

    def subspace(self, low_bounds, up_bounds, **kwargs):

        """subspace(self, lo_bounds, up_bounds)

        Build a sub space according to the actual Searchspace using two vectors containing lower and upper bounds of the subspace.
        Can change type to Constant if necessary

        Parameters
        ----------

        lo_bounds : list
            Lower bounds of the subspace. See `Variable` for more info.
        up_bounds : boolean, default=False
            Upper bounds of the subspace. See `Variable` for more info.

        Returns
        -------

        out : Searchspace
            Return a subspace of the actual Searchspace.

        Examples
        --------

        """

        sub = self.values.subset(low_bounds, up_bounds)
        sp = type(self)(sub, self.loss, **kwargs, **self.kwargs)

        return sp

    def show(self, X, Y, save=False, path=""):

        """show(X, Y, save=False, path="")

        Show solutions X associated to their values Y, according to the Searchspace

        Parameters
        ----------

        X : list[list[{int, float, str}, {int, float, str}...], ...]
            List of points to plot
        Y : list[{float, int}]
            Scores associated to X.

        Examples
        --------

        >>> import numpy as np
        >>> sp.show(sp.random_point(100), np.random.random(100))


        .. image:: ../_static/sp_show_ex.png
            :width: 924px
            :align: center
            :height: 487px
            :alt: alternate text
        """

        # if not isinstance(X, pd.DataFrame):
        #     X = pd.DataFrame(X, columns=[v.label for v in self.values])
        #
        # argmin = np.argmin(Y)
        # logger.info("Best individual")
        # logger.info(X.iloc[argmin, :])
        # logger.info(np.array(Y)[argmin])
        #
        # f, plots = plt.subplots(self.size, self.size, figsize=(19.2, 14.4))
        # f.suptitle("All evaluated solutions", fontsize=11)
        #
        # if len(X) < 100:
        #     s = 40
        # else:
        #     s = 10000 / len(X)
        #
        # if self.types[0] == "C":
        #     X.iloc[:, 0].value_counts().plot(kind="bar", ax=plots[0, 0])
        #     plots[0, 0].set_yticks([])
        #     plots[0, 0].xaxis.tick_top()
        #     plots[0, 0].tick_params(axis="x", labelsize=7 / len(self.types[0]))
        # else:
        #     plots[0, 0].hist(
        #         X.iloc[:, 0], 20, density=True, facecolor="g", alpha=0.75
        #     )
        #     plots[0, 0].set_yticks([])
        #     plots[0, 0].xaxis.tick_top()
        #     plots[0, 0].tick_params(axis="x", labelsize=7)
        #     plots[0, 0].set_xlim((self.values[0][0], self.values[0][1]))
        #
        # def onpick(event):
        #     ind = event.ind
        #     print("Selected point:\n", X.iloc[ind, :], Y[ind])
        #
        # for i in range(self.size):
        #
        #     if i > 0:
        #
        #         if self.types[i] == "C":
        #
        #             sorter = self.values[i]
        #             sorterIndex = dict(zip(sorter, range(len(sorter))))
        #
        #             new = (
        #                 X.iloc[:, i]
        #                 .value_counts()
        #                 .rename_axis("unique_values")
        #                 .reset_index(name="counts")
        #             )
        #             new["Rank"] = new["unique_values"].map(sorterIndex)
        #             new.sort_values("Rank", inplace=True)
        #             new.drop("Rank", 1, inplace=True)
        #             new = new.set_index("unique_values")
        #
        #             new["counts"].plot.barh(ax=plots[i, i], facecolor="g")
        #             plots[i, i].yaxis.tick_right()
        #             plots[i, i].tick_params(
        #                 axis="y", labelsize=7 / len(self.types[i])
        #             )
        #             plots[i, i].set_ylabel("")
        #
        #         else:
        #             plots[i, i].hist(
        #                 X.iloc[:, i],
        #                 20,
        #                 density=True,
        #                 facecolor="g",
        #                 alpha=0.75,
        #                 orientation="horizontal",
        #             )
        #             plots[i, i].yaxis.tick_right()
        #             plots[i, i].tick_params(axis="y", labelsize=7)
        #             plots[i, i].set_ylim((self.values[i][0], self.values[i][1]))
        #
        #     for j in range(i + 1, self.n_variables):
        #
        #         plots[i, j].axis("off")
        #
        #         if self.types[i] == "C" or self.types[j] == "C":
        #
        #             if self.types[i] == self.types[j]:
        #                 pass
        #             else:
        #                 if self.types[i] == "C":
        #                     idx = i
        #                     idx2 = j
        #                     vert = True
        #                 else:
        #                     idx = j
        #                     idx2 = i
        #                     vert = False
        #
        #                 data = []
        #                 for val in self.values[idx]:
        #                     data.append(
        #                         X.iloc[:, idx2].loc[X.iloc[:, idx] == val]
        #                     )
        #
        #                 plots[j, i].boxplot(
        #                     data,
        #                     vert=vert,
        #                     flierprops=dict(
        #                         marker="o",
        #                         markerfacecolor="green",
        #                         markersize=0.1,
        #                         markeredgecolor="green",
        #                     ),
        #                     labels=self.values[idx],
        #                 )
        #
        #         else:
        #             try:
        #                 plots[j, i].tricontourf(
        #                     X.iloc[:, i], X.iloc[:, j], Y, 10, cmap="Greys_r"
        #                 )
        #             except:
        #                 logger.warning("Triangularisation failed")
        #             plots[j, i].scatter(
        #                 X.iloc[:, i],
        #                 X.iloc[:, j],
        #                 c=Y,
        #                 s=s,
        #                 alpha=0.4,
        #                 cmap="coolwarm_r",
        #                 picker=True,
        #             )
        #             plots[j, i].set_xlim((self.values[i][0], self.values[i][1]))
        #             plots[j, i].set_ylim((self.values[j][0], self.values[j][1]))
        #
        #             plots[j, i].scatter(
        #                 X.iloc[argmin, i],
        #                 X.iloc[argmin, j],
        #                 c="cyan",
        #                 marker=(5, 2),
        #                 alpha=0.8,
        #                 s=150,
        #             )
        #
        #         if i == 0:
        #             plots[j, i].set_ylabel([v.label for v in self.values])
        #
        #         if j == self.n_variables - 1:
        #             plots[j, i].set_xlabel([v.label for v in self.values])
        #
        #         plots[j, i].set_xticks([])
        #         plots[j, i].set_yticks([])
        #
        # plt.subplots_adjust(
        #     left=0.050, bottom=0.050, right=0.970, top=0.970, wspace=0, hspace=0
        # )
        #
        # if save:
        #     save_path = os.path.join(path, f"matrix_sp.png")
        #
        #     plt.savefig(save_path, bbox_inches="tight")
        #     plt.close()
        # else:
        #     f.canvas.mpl_connect("pick_event", onpick)
        #     plt.show()
        #
        # # for i in range(self.n_variables):
        # #     if self.types[i] == "C":
        # #         inter = X[self.labels[i]].astype("category").cat.codes
        # #         X.drop(self.labels[i], axis=1)
        # #         X[self.labels[i]] = (inter - inter.min()) / (inter.max() - inter.min())
        # #     else:
        # #         X[self.labels[i]] = (X[self.labels[i]] - self.values[i][0]) / (self.values[i][1] - self.values[i][0])
        # #
        # # dataf = X.iloc[:, : self.n_variables]
        # # dataf["loss_value"] = Y
        # # parallel_coordinates(dataf, "loss_value", colormap="viridis_r")
        # # plt.show()
        pass


class HpoSearchspace(Searchspace):

    # Initialize the search space
    def __init__(self, values, loss, **kwargs):

        """__init__(values, loss, **kwargs)

        Parameters
        ----------

        values : ArrayVar
            Determines the bounds of the search space. For `HpoSearchspace` the
            `values` must be an `ArrayVar`. The :ref:`sp` will then manipulate
            this array.  Be carefull, in HPO paradigm, not all algorithms will
            manager `ArrayVar` composed of `Block` or other `ArrayVar`

        loss : LossFunc
            Callable of type `LossFunc`. See :ref:`lf` for more information.
            `loss` will be used by the :ref:`sp` object and by optimization
            algorithms.

        kwargs : dict
            Kwargs are the different addons one want to add to a `Variable`.
            Known addons are:
            * to_discrete : Converter
                * Will be called when converting to discrete is needed.
            * to_continuous : Converter
                * Will be called when converting to continuous is needed.
            * neighbor : Neighborhood
                * Will be called when a neighborhood is needed.
            * distance: Distance, default, Mixed
                * Will be called when computing a distance is needed


        """

        ##############
        # ASSERTIONS #
        ##############

        assert isinstance(values, ArrayVar), logger.error(
            f"`values` must be be an `ArrayVar`, got {values}"
        )

        self.distance = kwargs.pop("distance", Mixed())
        super(HpoSearchspace, self).__init__(values, loss, **kwargs)
        assert isinstance(self.distance, Distance), logger.error(
            f"Kwargs `distance` must be of type `Distance`, got {self.distance}"
        )
        self.distance.target = self


class ContinuousSearchspace(Searchspace):

    # Initialize the search space
    def __init__(self, values, loss, **kwargs):

        """__init__(values, loss, **kwargs)

        Parameters
        ----------

        values : ArrayVar
            Determines the bounds of the search space.
            For `ContinuousSearchspace` the `values` must be an `ArrayVar`
            of `FloatVar`.
            The :ref:`sp` will then manipulate this array.

        loss : LossFunc
            Callable of type `LossFunc`. See :ref:`lf` for more information.
            `loss` will be used by the :ref:`sp` object and by optimization
            algorithms.

        kwargs : dict
            Kwargs are the different addons one want to add to a `Variable`.
            Known addons are:
            * to_discrete : Converter
                * Will be called when converting to discrete is needed.
            * to_continuous : Converter
                * Will be called when converting to continuous is needed.
            * neighbor : Neighborhood
                * Will be called when a neighborhood is needed.
            * distance: Distance, default, Mixed
                * Will be called when computing a distance is needed

        """

        ##############
        # ASSERTIONS #
        ##############

        assert isinstance(values, ArrayVar) and all(
            isinstance(v, FloatVar) for v in values
        ), logger.error(
            f"`values` must be be an `ArrayVar` of `FloatVar`, got {values}"
        )

        self.distance = kwargs.pop("distance", Euclidean(self))
        assert isinstance(self.distance, Distance), logger.error(
            f"Kwargs `distance` must be of type `Distance`, got {self.distance}"
        )
        self.distance.target = self

        super(ContinuousSearchspace, self).__init__(values, loss, **kwargs)


class DiscreteSearchspace(Searchspace):

    # Initialize the search space
    def __init__(self, values, loss, **kwargs):

        """__init__(values, loss, **kwargs)

        Parameters
        ----------

        values : ArrayVar
            Determines the bounds of the search space.
            For `DiscreteSearchspace` the `values` must be an `ArrayVar`
            of `IntVar`.
            The :ref:`sp` will then manipulate this array.

        loss : LossFunc
            Callable of type `LossFunc`. See :ref:`lf` for more information.
            `loss` will be used by the :ref:`sp` object and by optimization
            algorithms.

        kwargs : dict
            Kwargs are the different addons one want to add to a `Variable`.
            Known addons are:
            * to_discrete : Converter
                * Will be called when converting to discrete is needed.
            * to_continuous : Converter
                * Will be called when converting to continuous is needed.
            * neighbor : Neighborhood
                * Will be called when a neighborhood is needed.
            * distance: Distance, default, Mixed
                * Will be called when computing a distance is needed

        """

        ##############
        # ASSERTIONS #
        ##############

        assert isinstance(values, ArrayVar) and all(
            isinstance(v, IntVar) for v in values
        ), logger.error(
            f"`values` must be be an `ArrayVar` of `FloatVar`, got {values}"
        )

        self.distance = kwargs.pop("distance", Euclidean(self))
        assert isinstance(self.distance, Distance), logger.error(
            f"Kwargs `distance` must be of type `Distance`, got {self.distance}"
        )
        self.distance.target = self

        super(ContinuousSearchspace, self).__init__(values, loss, **kwargs)
