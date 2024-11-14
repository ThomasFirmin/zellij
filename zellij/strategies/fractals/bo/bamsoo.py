# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)


from __future__ import annotations
from zellij.core.errors import InitializationError
from zellij.core.metaheuristic import UnitMetaheuristic, MonoObjective

from typing import Tuple, List, Optional, TYPE_CHECKING

from zellij.core.search_space import UnitSearchspace, BaseFractal

if TYPE_CHECKING:
    from zellij.strategies.fractals import Sampling
    from zellij.core.search_space import BaseFractal

import torch
import numpy as np

import gpytorch
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood


from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_mll
from botorch.exceptions import ModelFittingError


from collections import defaultdict

import logging

logger = logging.getLogger("zellij.BO")


class BaMSOO(UnitMetaheuristic, MonoObjective):
    """BaMSOO

    Bayesian optimization (BO) is a surrogate based optimization method which
    interpolates the actual loss function with a surrogate model, here a
    gaussian process.
    It is based on `BoTorch <https://botorch.org/>`_ and `GPyTorch <https://gpytorch.ai/>`__.

    Attributes
    ----------
    search_space : UnitSearchspace
        Search space object containing bounds of the search space
    tree_search : Tree_search
            Tree search algorithm applied on the partition tree.
    sampling : Sampling
        A :code:`Sampling` object.
    scoring : Scoring
        Function that defines how promising a space is according to sampled
        points.
    verbose : bool
        If False, there will be no print.
    surrogate : botorch.models.model.Model, default=SingleTaskGP
        Gaussian Process Regressor object from 'botorch'.
        Determines the surrogate model that Bayesian optimization will use to
        interpolate the loss function
    likelihood : gpytorch.mlls, default=ExactMarginalLogLikelihood
        gpytorch.mlls object it determines which MarginalLogLikelihood to use
        when optimizing kernel's hyperparameters
    acquisition : botorch.acquisition.acquisition.AcquisitionFunction, default = ExpectedImprovement
        An acquisition function or infill criteria, determines how 'promising'
        a point sampled from the surrogate is.
    initial_size : int, default=10
        Size of the initial set of solution to draw randomly.
    gpu: bool, default=True
        Use GPU if available
    kwargs
        Key word arguments linked to the surrogate and the acquisition function.

    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is
    :ref:`lf` : Describes what a loss function is in Zellij
    :ref:`sp` : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        search_space: UnitSearchspace,
        sampling: Sampling,
        nu: float,
        surrogate=SingleTaskGP,
        mll=ExactMarginalLogLikelihood,
        likelihood=GaussianLikelihood,
        initial_size: int = 10,
        gpu: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        """__init__

        Parameters
        ----------
        search_space : ContinuousSearchspace
            Search space object containing bounds of the search space
        verbose : bool
            If False, there will be no print.
        surrogate : botorch.models.model.Model, default=SingleTaskGP
            Gaussian Process Regressor object from 'botorch'.
            Determines the surrogate model that Bayesian optimization will use to
            interpolate the loss function
        mll : gpytorch.mlls, default=ExactMarginalLogLikelihood
            Object from gpytorch.mlls it determines which marginal loglikelihood to use
            when optimizing kernel's hyperparameters
        likelihood : gpytorch.likelihoods, default=GaussianLikelihood
            Object from gpytorch.likelihoods defining the likelihood.
        acquisition : botorch.acquisition.acquisition.AcquisitionFunction, default = ExpectedImprovement
            An acquisition function or infill criteria, determines how 'promising'
            a point sampled from the surrogate is.
        initial_size : int, default=10
            Size of the initial set of solution to draw randomly.
        gpu: bool, default=True
            Use GPU if available
        kwargs
            Key word arguments linked to the surrogate and the acquisition function.

        """

        super().__init__(search_space=search_space, verbose=verbose)

        ##################
        # DBA PARAMETERS #
        ##################

        self.sampling = sampling
        self.nu = nu

        #################
        # BO PARAMETERS #
        #################

        self.surrogate = surrogate
        self.mll = mll
        self.likelihood = likelihood
        self.initial_size = initial_size

        self.kwargs = kwargs

        ################
        # BO VARIABLES #
        ################

        self.cmll = None
        self.cmodel = None

        # Prior points
        self.train_x = torch.empty((0, self.search_space.size))
        self.train_obj = torch.empty((0, 1))
        self.state_dict = {}

        # Determine if BO is initialized or not
        self.initialized = False

        # Number of iterations
        self.iterations = 0

        if gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                logger.warning(f"No GPU available for BaMSOO.")
        else:
            self.device = torch.device("cpu")

        self.dtype = torch.double
        self._build_kwargs()

        #################
        # DBA VARIABLES #
        #################

        self.vmax = float("inf")
        self.best_score = float("inf")
        self.N = 1
        self.current_level = 0
        self.current_maxlevel = 0

        # Children fractals
        self.children = []
        self.child = None
        self._fidx = []

        self.tree = defaultdict(list)
        self.tree_score = defaultdict(list)
        self.tree[0].append(self.search_space)
        self.tree_score[0].append(0)

    @property
    def search_space(self) -> BaseFractal:
        return self._search_space

    @search_space.setter
    def search_space(self, value: BaseFractal):
        if isinstance(value, BaseFractal):
            self._search_space = value
        else:
            raise InitializationError(
                f"Searchspace in DBA must be of type BaseFractal. Got {type(value).__name__}."
            )

    @property
    def sampling(self) -> Sampling:
        return self._sampling

    @sampling.setter
    def sampling(self, value: Sampling):
        if value:  # If there is sampling
            self._sampling = value
        else:
            raise InitializationError(f"DBA must implement at least an exploration.")

    def _sample(
        self,
        sp: BaseFractal,
        X: Optional[list],
        Y: Optional[np.ndarray],
        constraint: Optional[np.ndarray] = None,
        info: Optional[np.ndarray] = None,
        xinfo: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict, dict]:
        self.sampling.reset()
        self.sampling.search_space = sp
        points, info_dict, xinfo_dict = self.sampling.forward(
            X, Y, constraint, info, xinfo
        )
        if len(points) > 0:
            return points, info_dict, xinfo_dict  # Continue exploration
        else:
            return [], {"algorithm": "EndSample"}, {}  # Exploration ending

    def _initialize_model(
        self,
        train_x: torch.Tensor,
        train_obj: torch.Tensor,
        state_dict: Optional[dict] = None,
    ):
        train_x.to(self.device, dtype=self.dtype)
        train_obj.to(self.device, dtype=self.dtype)

        likelihood = self.likelihood(**self.likelihood_kwargs)

        # define models for objective and constraint
        model = self.surrogate(
            train_x,
            train_obj,
            likelihood=likelihood,
            **self.model_kwargs,
        )

        mll = self.mll(
            model.likelihood,
            model,
            **self.mll_kwargs,
        )

        # load state dict if it is passed
        if state_dict:
            model.load_state_dict(state_dict)

        model.to(self.device)

        return mll, model

    def get_posterior(self, X) -> Tuple[float, float]:
        if self.cmodel is not None:
            posterior = self.cmodel.posterior(X=X)
            mean = posterior.mean.squeeze(-2).squeeze(-1)
            sigma = posterior.variance.clamp_min(1e-12).sqrt().view(mean.shape)

            return mean.item(), sigma.item()
        else:
            return float("inf"), float("inf")

    def _build_kwargs(self):
        # Surrogate kwargs
        self.model_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.surrogate.__init__.__code__.co_varnames
        }

        for m in self.model_kwargs.values():
            if isinstance(m, torch.nn.Module):
                m.to(self.device)

        # Likelihood kwargs
        self.likelihood_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.likelihood.__init__.__code__.co_varnames
        }
        for m in self.likelihood_kwargs.values():
            if isinstance(m, torch.nn.Module):
                m.to(self.device)

        # MLL kwargs
        self.mll_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.mll.__init__.__code__.co_varnames
        }
        for m in self.mll_kwargs.values():
            if isinstance(m, torch.nn.Module):
                m.to(self.device)

        logger.debug(self.model_kwargs, self.likelihood_kwargs, self.mll_kwargs)

    def reset(self):
        """reset()

        reset :code:`Bayesian_optimization` to its initial state.

        """
        self.initialized = False
        self.train_x = torch.empty((0, self.search_space.size))
        self.train_obj = torch.empty((0, 1))
        self.state_dict = {}

    def _get_children(self) -> List[BaseFractal]:
        scr = self.tree_score[self.current_level]
        if len(scr) == 0:
            self.current_level = self.current_level % self.current_maxlevel
            if self.current_level == 0:
                self.vmax = float("inf")
                self.current_level = 1
            else:
                self.current_level += 1
            return self._get_children()
        else:
            argmin = np.argmin(scr)
            if scr[argmin] < self.vmax:
                parent = self.tree[self.current_level].pop(argmin)
                self.vmax = self.tree_score[self.current_level].pop(argmin)
                children = parent.create_children()
                return children
            else:
                self.current_level = self.current_level % self.current_maxlevel
                if self.current_level == 0:
                    self.vmax = float("inf")
                    self.current_level = 1
                else:
                    self.current_level += 1
                return self._get_children()

    def forward(
        self,
        X: Optional[list],
        Y: Optional[np.ndarray],
        constraint: Optional[np.ndarray],
        info: Optional[np.ndarray],
        xinfo: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict, dict]:
        """forward

        Abstract method describing one step of the :ref:`meta`.

        Parameters
        ----------
        X : list
            List of points.
        Y : numpy.ndarray[float]
            List of loss values.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Dictionnary of additionnal information linked to :code:`points`.
        """

        self.iterations += 1
        self.current_maxlevel = len(self.tree) + 1

        if X is not None and Y is not None and self.child is not None:

            for x, y in zip(X, Y):
                self.child.add_solutions(x, y)

            self.child.evaluated = True
            self.child.score = self.child.best_loss
            self.tree[self.child.level].append(self.child)
            self.tree_score[self.child.level].append(self.child.score)
            if self.child.score < self.best_score:
                self.best_score = self.child.score

            npx = np.array(X)
            npy = np.squeeze(Y)
            mask = np.isfinite(npy)

            mx = npx[mask][0]
            my = npy[mask]

            if len(my) > 0:
                new_x = torch.tensor(mx, dtype=self.dtype)
                new_obj = torch.tensor(my, dtype=self.dtype).unsqueeze(-1)

                # update training points
                self.train_x = torch.cat([self.train_x, new_x])
                self.train_obj = torch.cat([self.train_obj, new_obj])

                # reinitialize the models so they are ready for fitting on next iteration
                # use the current state dict to speed up fitting
                self.cmll, self.cmodel = self._initialize_model(
                    self.train_x,
                    self.train_obj,
                    self.state_dict,
                )

                self.state_dict = self.cmodel.state_dict()
                try:
                    with gpytorch.settings.max_cholesky_size(300):
                        # run N_BATCH rounds of BayesOpt after the initial random batch
                        # fit the models
                        fit_gpytorch_mll(self.cmll)

                except ModelFittingError:
                    logger.warning("ModelFittingError, previous model will be used.")
            else:
                logger.warning(
                    "InputError, Y are not finite. Previous model will be used"
                )

        if len(self.children) == 0:
            self.children = self._get_children()

        self.child = self.children.pop()
        print(f"LEVEL : {self.child.level}")

        self.N += 1

        points, info_dict, xinfo_dict = self._sample(
            self.child, X, Y, constraint, info, xinfo
        )

        if len(points) == 0:
            return self.forward(None, None, None, None, None)

        if self.cmodel is None:
            xinfo_dict["mean"] = float("inf")
            xinfo_dict["std"] = float("inf")
            xinfo_dict["beta"] = float("inf")
            return points, info_dict, xinfo_dict
        else:
            x = torch.FloatTensor(points).to(self.device)
            mean, std = self.get_posterior(x)

        beta = np.sqrt(2 * np.log(np.pi**2 * self.N**2 / (6 * self.nu)))
        right_part = beta * std
        if mean - right_part <= self.best_score:
            xinfo_dict["mean"] = mean
            xinfo_dict["std"] = std
            xinfo_dict["beta"] = beta
            return points, info_dict, xinfo_dict
        else:
            self.child.score = mean + right_part
            self.tree[self.child.level].append(self.child)
            self.tree_score[self.child.level].append(self.child.score)
            if self.child.score < self.best_score:
                self.best_score = self.child.score

            return self.forward(None, None, None, None, None)
