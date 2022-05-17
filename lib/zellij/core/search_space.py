# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   ThomasFirmin
# @Last modified time: 2022-05-03T16:00:13+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
from abc import ABC, abstractmethod


import logging

logger = logging.getLogger("zellij.space")


class Searchspace(ABC):
    def __init__(self, values):
        pass


# TYPES: REAL: R, DISCRETE: D, CATEGORICAL: C, CONSTANT: K
class Mixed:

    """Searchspace

    Searchspace is an essential class for Zellij. Define your search space with this object.

    Attributes
    ----------

    labels : list[str]
        Labels associated to each dimension of the search space
    types : list[{'R','D','C','K'}]
        Types associated to each dimension of the search space.

        * R : Real dimension (e.g. float)
        * D : Discrete dimension (e.g. int)
        * C : Categorical dimension (e.g. {'dog', 'cats', 'rabbit'})
        * K : Constant dimension, used to define an unvariable dimension.

         Used in space decomposition when a dimension cannot be decomposed because it became too small (e.g. discrete).\
         For a constant value, implementing it directly to the loss function is preferable.

    values : list[{[float, float], [int, int], [{str, int, float}, {str, int, float}...]}]
        Bounds associated to each dimension.

        * For R, D types, values must be of form [a:{float,int}, b:{float,int}] with a < b.
        * For C type, values must be of form [{str, int, float}, {str, int, float}...]
        * For K type, values can be of any type because it will not be used and only passed to the loss function.

    neighborhood : list[{float, int, -1}]
        Neighborhood  of a solution for each dimension. For a solution :math:`x=[v_1, v_2]`, bounds of the neighborhood of :math:`v_i` of :math:`x` are computed by :math:`[max(v_i-n, lower bound), min(v_i+n, upper bound)]`,\
        except for C dimension, a neighbor is computed by drawing a random value assoiated to the dimension. For K dimension there is no neighborhood.

        * For R type, the neighborhood can be a float, or an int.
        * For D type, the neighborhood is an int.
        * For C,K types, the neighborhood must be -1.


    sub_values : list[{[float, float], [int, int], [{str, int, float}, {str, int, float}...]}]
        When building a subspace. Original values are saved in sub_values, so it can easily convert a subspace into a mixed or continuous search space. See values
    n_variables : list
        Number of dimensions
    k_index : list[int]
        Indexes of constant dimensions.

    Methods
    -------
    random_attribute(self,size=1,replace=True, exclude=None)
        Draw random features from the search space. Features of type K, cannot be drawn (i.e. their probability are set to 0).

    random_value(self, attribute, size=1, replace = True, exclude=None)
        Draw random values of an attribute from the search space, using uniform distribution. Features of type K return their constant value.

    random_point(self,size=1)
        Return a random point from the search space

    convert_to_continuous(self,points,reverse=False,sub_values=False)
        Convert given points from mixed to continuous, or, from continuous to mixed.

    general_convert(self)
        Convert the search space by building a continuous one.

    subspace(self,lo_bounds,up_bounds)
        Build a sub space according to the actual Searchspace using two vectors containing lower and upper bounds of the subspace.

    show(self,X,Y)
        Show solutions X associated to their values Y, according to the Searchspace

    See Also
    --------
    LossFunc : Parent class for a loss function.

    Examples
    --------
    >>> from zellij.core.search_space import Searchspace
    >>> labels = ["learning rate","neurons","activation"]
    >>> types = ["R","D","C"]
    >>> values = [[-5.0, 5.0],[0, 20],["relu","tanh","sigmoid"]]
    >>> sp = Mixed(labels,types,values, neighborhood)

    """

    # Initialize the search space
    def __init__(self, labels, types, values, neighborhood=0.10):

        """__init__(self, labels, types, values, neighborhood = 0.10 )

        Parameters
        ----------
        label : list[str]
            Labels associated to each dimension of the search space
        type : list[{'R','D','C','K'}]
            Types associated to each dimension of the search space.
            R : Real dimension (e.g. float)
            D : Discrete dimension (e.g. int)
            C : Categorical dimension (e.g. {'dog', 'cats', 'rabbit'})
            K : Constant dimension, used to define an unvariable dimension. \
             Used in space decomposition when a dimension cannot be decomposed because it became too small (e.g. discrete).\
             For a constant value, implementing it directly to the loss function is preferable.

        values : list[{[float, float], [int, int], [{str, int, float}, {str, int, float}...]}]
            Bounds associated to each dimension.\
            For R, D types, values must be of form [a:{float,int}, b:{float,int}] with a < b.\
            For C type, values must be of form [{str, int, float}, {str, int, float}...]
            For K type, values can be of any type because it will not be used and only passed to the loss function.

        neighborhood : {float, list[{float, int, -1}]}, default=0.10
            Neighborhood  of a solution for each dimension.\
            For a solution x=[v1, v2], bounds of the neighborhood of vi of x are computed by [max(vi-n, lower bound), min(vi+n, upper bound)],\
            except for C dimension, a neighbor is computed by drawing a random value assoiated to the dimension. For K dimension there is no neighborhood.

            If a float, N, between 0 and 1 is given. The neighborhood will be computed according to precedent rules, n = N*(upper bounds - lower bounds).
            The neighborhood for R, D dimension will be a percentage of the dimension size.

            You can manually determine the neighborhood according to following rules:
                - neighborhood must be of type list[{float, int, -1}], each value of the list determine the type of neighborhood for each dimension:
                    - For R type, the neighborhood can be a float, or an int.
                    - For D type, the neighborhood is an int.
                    - For C,K types, the neighborhood must be -1.
        """

        ##############
        # ASSERTIONS #
        ##############

        valid_types = ["R", "D", "C", "K"]

        assert len(labels) > 0, "Labels length must be > 0"
        assert len(types) > 0, "Type length must be > 0"
        assert len(values) > 0, "Values length must be > 0"
        assert (
            len(labels) == len(types) == len(values)
        ), "Labels, types and values must have the same length"

        assert all(isinstance(l, str) for l in labels), "Labels must be strings"
        assert all(
            t in valid_types for t in types
        ), "Types must be equal to 'R', 'D', 'C' or 'K' "

        for v, t in zip(values, types):
            if t == "R":
                assert (
                    isinstance(v, list)
                    and len(v) == 2
                    and (isinstance(v[0], float) or isinstance(v[0], int))
                    and (isinstance(v[1], float) or isinstance(v[1], int))
                    and v[0] < v[1]
                ), f"Values of type 'R' must be of the form [a : int, b : int] or [a : float, b : float] and b > a, got {v}"

            elif t == "D":
                assert (
                    isinstance(v, list)
                    and len(v) == 2
                    and isinstance(v[0], int)
                    and isinstance(v[1], int)
                    and v[0] < v[1]
                ), f"Values of type 'D' must be of the form [a : int, b : int] and b > a, got {v}"

            elif t == "C":
                assert (
                    isinstance(v, list) and len(v) >= 2
                ), f"Values of type 'C' must be of the form [value 1, value 2,...], got {v}"

        ##############
        # PARAMETERS #
        ##############

        self.labels = labels
        self.types = types
        self.values = values
        self.sub_values = None

        #############
        # VARIABLES #
        #############

        self.n_variables = len(labels)

        self.k_index = [i for i, x in enumerate(self.types) if x == "K"]

    # Add addon in addons :)
    def _add_addon(self, addon):
        key = f"{addon.__class__.__bases__[0].__name__}".lower()
        if hasattr(self, key):
            logger.warning(f"A {key} already is already implemented")
            logger.warning(f"{key} will be overwritten")
        setattr(self, key, addon)

    # Return 1 or size=n random attribute from the search space, can exclude one attribute
    def random_attribute(self, size=1, replace=True, exclude=None):

        """random_attribute(size=1,replace=True, exclude=None)

        Draw random features from the search space. Features of type K, cannot be drawn (i.e. their probability are set to 0).

        Parameters
        ----------
        size : int, default=1
            Select randomly <size> features.
        replace : boolean, default=True
            Select randomly <size> features with replacement if True, without else.
            See numpy.random.choice
        exclude : str, default=None
            Exclude dimension of label==<exclude>.

        Returns
        -------

        _ : numpy.array(dtype=int)
            Array of selected dimension index.

        Examples
        --------
        >>> rand_att = sp.random_attribute(5)
        >>> print(f"Random Attributes: {rand_att}")
        Random Attributes: ['learning rate' 'activation' 'neurons' 'activation' 'activation']
        """

        try:
            index = self.labels.index(exclude)
            p = np.full(
                self.n_variables,
                1 / (self.n_variables - (len(self.k_index) + 1)),
            )
            p[index] = 0

        except ValueError:
            p = np.full(
                self.n_variables, 1 / (self.n_variables - len(self.k_index))
            )

        for i in self.k_index:
            p[i] = 0

        return np.random.choice(self.labels, size=size, replace=replace, p=p)

    # Return a or size=n random value from an attribute, can exclude one value
    def random_value(self, attribute, size=1, replace=True, exclude=None):

        """random_value(attribute, size=1, replace = True, exclude=None)

        Draw random values of an attribute from the search space, using uniform distribution. Features of type K return their constant value.

        Parameters
        ----------
        attribute : str
            Select dimension of label==<exclude> from which to draw a random value.
        size : int, default=1
            Select randomly <size> features.
        replace : boolean, default=True
            Select randomly <size> features with replacement if True, without else.
            See numpy.random.choice
        exclude : str, default=None
            Exclude dimension of label==<exclude>.

        Returns
        -------

        _ : numpy.array(dtype={int, float})
            Array of floats for R dimension, ints for D, indexes (int) for C.

        Examples
        --------

        Real values

        >>> rand_val = sp.random_value('learning rate',5)
        >>> print(f"Random 'learning rate': {rand_val}")
        Random 'learning rate': [-2.137716545134689, 2.088492759876778, 3.970466658083497, -3.665643582672203, 3.116564115026975]

        Discrete values

        >>> rand_val = sp.random_value('neurons',5)
        >>> print(f"Random 'neurons': {rand_val}")
        Random 'neurons': [3, 8, 15, 8, 5]

        Categorical values

        >>> rand_val = sp.random_value('activation',5)
        >>> print(f"Random 'activation': {rand_val}")
        Random 'activation': ['tanh' 'sigmoid' 'tanh' 'sigmoid' 'sigmoid']

        """

        index = self.labels.index(attribute)

        if self.types[index] == "R":
            return np.random.uniform(
                self.values[index][0], self.values[index][1], size=size
            ).tolist()

        elif self.types[index] == "D":
            return np.random.randint(
                self.values[index][0], self.values[index][1] + 1, size=size
            ).tolist()

        elif self.types[index] == "C":

            try:
                idx = self.values[index].index(exclude)
                p = np.full(
                    len(self.values[index]), 1 / (len(self.values[index]) - 1)
                )
                p[idx] = 0
            except ValueError:
                p = np.full(
                    len(self.values[index]), 1 / len(self.values[index])
                )

            return np.random.choice(
                self.values[index], size=size, replace=replace, p=p
            )

        else:
            return [self.values[index] for _ in range(size)]

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

        points = []

        for i in range(size):
            new_point = []
            for l in self.labels:
                new_point.append(self.random_value(l)[0])
            points.append(new_point[:])

        return points

    def general_convert(self):

        """general_convert()

        Convert the search space by building a continuous one.
        labels are identical, all types are converted to R, and all bounds are between [0,1].
        Build an adptated neighborhood

        Returns
        -------

        sp : Searchspace
            Continuous Searchspace.

        """

        label = self.labels[:]
        type = ["R"] * self.n_variables
        values = [[0, 1]] * self.n_variables
        neighborhood = []

        for att in range(self.n_variables):

            if self.types[att] == "R" or self.types[att] == "D":

                neighborhood.append(
                    self.neighborhood[att]
                    / (self.values[att][1] - self.values[att][0])
                )

            else:

                neighborhood.append(1)

        sp = Searchspace(label, type, values, neighborhood)

        return sp

    def subspace(self, lo_bounds, up_bounds):

        """convert_to_continuous(points,reverse=False,sub_values=False)

        Build a sub space according to the actual Searchspace using two vectors containing lower and upper bounds of the subspace.
        Transforms types to K if necessary.
        Builds an adaptated neighborhood to avoid large neighborhood compare to the subspace size.
        Categorical lower and upper bounds of the subspace are determined according to a slice of the vector containing values:

            * Original: ["dog", "cat", "rabbit", "horse"]

            * lo_bounds = ["dog"]

            * up_bounds = ["rabbit"]

            * Subspace: ["dog", "cat", "rabbit"]

        Parameters
        ----------

        lo_bounds : {list[list[{int, float, str}, {int, float, str}...], ...], list[list[float, float...], ...]}
            List of points to convert
        up_bounds : boolean, default=False
            If False convert points from mixed to continuous, if True, from continuous to mixed

        Returns
        -------

        subspace : Searchspace
            Return a subspace of the actual Searchspace.

        """

        new_values = []
        new_neighborhood = []
        new_types = self.types[:]

        for i in range(len(lo_bounds)):

            if lo_bounds[i] != up_bounds[i]:
                if self.types[i] == "R":
                    new_values.append(
                        [
                            float(np.max([lo_bounds[i], self.values[i][0]])),
                            float(np.min([up_bounds[i], self.values[i][1]])),
                        ]
                    )
                    new_neighborhood.append(
                        self.neighborhood[i]
                        * (new_values[-1][1] - new_values[-1][0])
                        / (self.values[i][1] - self.values[i][0])
                    )

                elif self.types[i] == "D":
                    new_values.append(
                        [
                            int(np.max([lo_bounds[i], self.values[i][0]])),
                            int(np.min([up_bounds[i], self.values[i][1]])),
                        ]
                    )
                    n_val = int(
                        self.neighborhood[i]
                        * (new_values[-1][1] - new_values[-1][0])
                        / (self.values[i][1] - self.values[i][0])
                    )
                    new_neighborhood.append(1 if n_val == 0 else n_val)

                elif self.types[i] == "C":
                    new_neighborhood.append(-1)

                    lo_idx = self.values[i].index(lo_bounds[i])
                    up_idx = self.values[i].index(up_bounds[i])

                    if lo_idx > up_idx:
                        inter = up_idx
                        up_idx = lo_idx
                        lo_idx = inter

                    new_values.append(self.values[i][lo_idx : up_idx + 1])

                elif self.types[i] == "K":
                    new_neighborhood.append(-1)
                    new_values.append(self.values[i])

            # If lower and upper bounds are equal create a constant
            else:
                if self.types[i] != "K":
                    new_values.append(lo_bounds[i])
                    new_types[i] = "K"
                    new_neighborhood.append(-1)
                else:
                    new_neighborhood.append(-1)
                    new_values.append(self.values[i])

        subspace = Searchspace(
            self.labels, new_types, new_values, new_neighborhood
        )

        if subspace.sub_values == None:
            subspace.sub_values = self.values
        else:
            subspace.sub_values = self.sub_values

        return subspace

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

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.labels)

        argmin = np.argmin(Y)
        logger.info("Best individual")
        logger.info(X.iloc[argmin, :])
        logger.info(np.array(Y)[argmin])

        f, plots = plt.subplots(
            self.n_variables, self.n_variables, figsize=(19.2, 14.4)
        )
        f.suptitle("All evaluated solutions", fontsize=11)

        if len(X) < 100:
            s = 40
        else:
            s = 10000 / len(X)

        if self.types[0] == "C":
            X.iloc[:, 0].value_counts().plot(kind="bar", ax=plots[0, 0])
            plots[0, 0].set_yticks([])
            plots[0, 0].xaxis.tick_top()
            plots[0, 0].tick_params(axis="x", labelsize=7 / len(self.types[0]))
        else:
            plots[0, 0].hist(
                X.iloc[:, 0], 20, density=True, facecolor="g", alpha=0.75
            )
            plots[0, 0].set_yticks([])
            plots[0, 0].xaxis.tick_top()
            plots[0, 0].tick_params(axis="x", labelsize=7)
            plots[0, 0].set_xlim((self.values[0][0], self.values[0][1]))

        def onpick(event):
            ind = event.ind
            print("Selected point:\n", X.iloc[ind, :], Y[ind])

        for i in range(self.n_variables):

            if i > 0:

                if self.types[i] == "C":

                    sorter = self.values[i]
                    sorterIndex = dict(zip(sorter, range(len(sorter))))

                    new = (
                        X.iloc[:, i]
                        .value_counts()
                        .rename_axis("unique_values")
                        .reset_index(name="counts")
                    )
                    new["Rank"] = new["unique_values"].map(sorterIndex)
                    new.sort_values("Rank", inplace=True)
                    new.drop("Rank", 1, inplace=True)
                    new = new.set_index("unique_values")

                    new["counts"].plot.barh(ax=plots[i, i], facecolor="g")
                    plots[i, i].yaxis.tick_right()
                    plots[i, i].tick_params(
                        axis="y", labelsize=7 / len(self.types[i])
                    )
                    plots[i, i].set_ylabel("")

                else:
                    plots[i, i].hist(
                        X.iloc[:, i],
                        20,
                        density=True,
                        facecolor="g",
                        alpha=0.75,
                        orientation="horizontal",
                    )
                    plots[i, i].yaxis.tick_right()
                    plots[i, i].tick_params(axis="y", labelsize=7)
                    plots[i, i].set_ylim((self.values[i][0], self.values[i][1]))

            for j in range(i + 1, self.n_variables):

                plots[i, j].axis("off")

                if self.types[i] == "C" or self.types[j] == "C":

                    if self.types[i] == self.types[j]:
                        pass
                    else:
                        if self.types[i] == "C":
                            idx = i
                            idx2 = j
                            vert = True
                        else:
                            idx = j
                            idx2 = i
                            vert = False

                        data = []
                        for val in self.values[idx]:
                            data.append(
                                X.iloc[:, idx2].loc[X.iloc[:, idx] == val]
                            )

                        plots[j, i].boxplot(
                            data,
                            vert=vert,
                            flierprops=dict(
                                marker="o",
                                markerfacecolor="green",
                                markersize=0.1,
                                markeredgecolor="green",
                            ),
                            labels=self.values[idx],
                        )

                else:
                    try:
                        plots[j, i].tricontourf(
                            X.iloc[:, i], X.iloc[:, j], Y, 10, cmap="Greys_r"
                        )
                    except:
                        logger.warning("Triangularisation failed")
                    plots[j, i].scatter(
                        X.iloc[:, i],
                        X.iloc[:, j],
                        c=Y,
                        s=s,
                        alpha=0.4,
                        cmap="coolwarm_r",
                        picker=True,
                    )
                    plots[j, i].set_xlim((self.values[i][0], self.values[i][1]))
                    plots[j, i].set_ylim((self.values[j][0], self.values[j][1]))

                    plots[j, i].scatter(
                        X.iloc[argmin, i],
                        X.iloc[argmin, j],
                        c="cyan",
                        marker=(5, 2),
                        alpha=0.8,
                        s=150,
                    )

                if i == 0:
                    plots[j, i].set_ylabel(self.labels[j])

                if j == self.n_variables - 1:
                    plots[j, i].set_xlabel(self.labels[i])

                plots[j, i].set_xticks([])
                plots[j, i].set_yticks([])

        plt.subplots_adjust(
            left=0.050, bottom=0.050, right=0.970, top=0.970, wspace=0, hspace=0
        )

        if save:
            save_path = os.path.join(path, f"matrix_sp.png")

            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            f.canvas.mpl_connect("pick_event", onpick)
            plt.show()

        # for i in range(self.n_variables):
        #     if self.types[i] == "C":
        #         inter = X[self.labels[i]].astype("category").cat.codes
        #         X.drop(self.labels[i], axis=1)
        #         X[self.labels[i]] = (inter - inter.min()) / (inter.max() - inter.min())
        #     else:
        #         X[self.labels[i]] = (X[self.labels[i]] - self.values[i][0]) / (self.values[i][1] - self.values[i][0])
        #
        # dataf = X.iloc[:, : self.n_variables]
        # dataf["loss_value"] = Y
        # parallel_coordinates(dataf, "loss_value", colormap="viridis_r")
        # plt.show()
