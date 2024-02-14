# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)


from __future__ import annotations
from zellij.core.errors import InputError
from zellij.core.metaheuristic import UnitMetaheuristic

from abc import abstractmethod, ABC
from typing import Tuple, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.search_space import UnitSearchspace

import torch
import numpy as np
import time

import gpytorch

from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood

from botorch.models import SingleTaskGP
from botorch.acquisition.analytic import (
    AnalyticAcquisitionFunction,
    ExpectedImprovement,
)
from botorch.utils import standardize
from botorch.fit import fit_gpytorch_mll
from botorch.exceptions import ModelFittingError
from botorch.optim import optimize_acqf
from botorch.models.transforms import Log
from botorch.models.transforms.outcome import Standardize

from torch.quasirandom import SobolEngine

import logging

logger = logging.getLogger("zellij.BO")


class CostModel(torch.nn.Module, ABC):
    """
    Simple abstract class for a cost model.
    """

    @abstractmethod
    def forward(self, X):
        pass


class CostModelGP(CostModel):
    """
    A basic cost model that assumes the cost is positive.
    It models the log cost to guarantee positive cost predictions.
    """

    def __init__(self, X, Y_cost):
        assert torch.all(Y_cost > 0)
        super().__init__()
        gp = SingleTaskGP(
            train_X=X,
            train_Y=Y_cost,
            outcome_transform=Log(),
        )
        mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
        fit_gpytorch_mll(mll)
        self.gp = gp

    def forward(self, X):
        return torch.exp(self.gp(X).mean)


class AcquisitionWithCost(AnalyticAcquisitionFunction):
    """
    This is the acquisition function EI(x) / c(x) ^ alpha, where alpha is a decay
    factor that reduces or increases the emphasis of the cost model c(x).
    """

    def __init__(self, acquisition, model, cost_model, alpha: float = 1, **kwargs):
        super().__init__(model=model)
        self.model = model
        self.cost_model = cost_model
        self.acquisition = acquisition(model=model, **kwargs)
        self.alpha = alpha

    def forward(self, X):
        acq = self.acquisition(X)
        cost = torch.pow(self.cost_model(X)[:, 0], self.alpha)
        print(f"ACQ/COST: {acq}/{cost}")
        return acq / cost


class CostAwareBO(UnitMetaheuristic):
    """CostAwareBO

    Cost-Aware Bayesian Optimization

    It is based on `BoTorch <https://botorch.org/>`_ and `GPyTorch <https://gpytorch.ai/>`__.

    Attributes
    ----------
    search_space : Searchspace
        Search space object containing bounds of the search space
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
        budget: float,
        verbose: bool = True,
        surrogate=SingleTaskGP,
        mll=ExactMarginalLogLikelihood,
        likelihood=GaussianLikelihood,
        acquisition=ExpectedImprovement,
        n_candidates: Optional[int] = None,
        initial_size: int = 10,
        cholesky_size: int = 800,
        gpu: bool = False,
        time_budget=False,
        **kwargs,
    ):
        """__init__

        Parameters
        ----------
        search_space : UnitSearchspace
            Search space object containing bounds of the search space
        budget : float
            Total budget allocated to the algorithm
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
        n_canditates : int, default=None
            Number of candidates to sample with the surrogate.
        initial_size : int, default=10
            Size of the initial set of solution to draw randomly.
        cholesky_size : int, default=800
            Maximum size for which Lanczos method is used instead of Cholesky decomposition.
        gpu: bool, default=True
            Use GPU if available
        time_budget: bool, default=False
            If True, then budget is considered as time in seconds.
        kwargs
            Key word arguments linked to the surrogate, mll, likelihood or acquisition.
        """

        super().__init__(search_space, verbose)

        ##############
        # PARAMETERS #
        ##############

        self.budget = budget
        self.time_budget = time_budget
        self.start_time = time.time()

        self.surrogate = surrogate
        self.mll = mll
        self.likelihood = likelihood
        self.acquisition = acquisition

        self.n_candidates = n_candidates
        self.initial_size = initial_size

        self.kwargs = kwargs

        #############
        # VARIABLES #
        #############
        # Determine if BO is initialized or not
        self.initialized = False

        # Number of iterations
        self.iterations = 0

        if gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        self.dtype = torch.double

        # Prior points
        self.train_x = torch.empty(
            (0, self.search_space.size), dtype=self.dtype, device=self.device
        )
        # Prior objective
        self.train_obj = torch.empty((0, 1), dtype=self.dtype, device=self.device)
        # Cost objective
        self.train_c = torch.empty((0, 1), dtype=self.dtype, device=self.device)

        self.sobol = SobolEngine(dimension=self.search_space.size, scramble=True)

        self._build_kwargs()

        # Count generated models
        self.models_number = 0

        self.cholesky_size = cholesky_size

        self.iterations = 0

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

        self.acqf_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.acquisition.__init__.__code__.co_varnames
        }
        for m in self.acqf_kwargs.values():
            if isinstance(m, torch.nn.Module):
                m.to(self.device)

        logger.debug(self.model_kwargs, self.likelihood_kwargs, self.mll_kwargs)

    def _generate_initial_data(self) -> List[list]:
        return self.search_space.random_point(self.initial_size)

    # Initialize a surrogate
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
            outcome_transform=Standardize(m=1),
            **self.model_kwargs,
        )
        model.to(self.device)

        if "num_data" in self.mll.__init__.__code__.co_varnames:
            mll = self.mll(
                model.likelihood,
                model.model,
                num_data=train_x.shape[-2],  # type: ignore
                **self.mll_kwargs,
            )
        else:
            mll = self.mll(
                model.likelihood,
                model,
                **self.mll_kwargs,
            )

        # load state dict if it is passed
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return mll, model

    def _optimize_acqf_and_get_observation(
        self, acq_func, restarts: int = 10, raw: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimizes the acquisition function, and returns a new candidate
        and a noisy observation."""

        lower = torch.tensor(
            self.search_space.lower, device=self.device, dtype=self.dtype
        )
        upper = torch.tensor(
            self.search_space.upper, device=self.device, dtype=self.dtype
        )

        bounds = torch.stack((lower, upper))

        # optimize
        candidates, acqf_value = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=self.kwargs.get("q", 1),
            num_restarts=restarts,
            raw_samples=raw,  # used for intialization heuristic
            options={
                "batch_limit": self.kwargs.get("batch_limit", 5),
                "maxiter": self.kwargs.get("maxiter", 200),
            },
        )

        return candidates.detach(), acqf_value

    def reset(self):
        """reset()

        reset :code:`Bayesian_optimization` to its initial state.

        """
        self.initialized = False
        self.train_x = torch.empty(
            (0, self.search_space.size), dtype=self.dtype, device=self.device
        )
        self.train_obj = torch.empty((0, 1), dtype=self.dtype, device=self.device)
        self.train_c = torch.empty((0, 1), dtype=self.dtype, device=self.device)

    def forward(
        self,
        X: Optional[list],
        Y: Optional[np.ndarray],
        secondary: Optional[np.ndarray],
        constraint: Optional[np.ndarray],
    ) -> Tuple[List[list], dict]:
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

        if not self.initialized:
            # call helper functions to generate initial training data and initialize model
            train_x = self._generate_initial_data()
            self.initialized = True
            return train_x, {
                "iteration": self.iterations,
                "algorithm": "InitCArBO",
                "model": -1,
            }
        elif X is None or Y is None or secondary is None:
            raise InputError(
                "After initialization Bayesian optimization must receive non-empty X, Y and secondary in forward."
            )
        else:
            torch.cuda.empty_cache()
            self.iterations += 1

            new_x = torch.tensor(X, dtype=self.dtype, device=self.device)
            new_obj = torch.tensor(Y, dtype=self.dtype, device=self.device).unsqueeze(
                -1
            )
            new_c = torch.tensor(
                secondary[:, 0], dtype=self.dtype, device=self.device
            ).unsqueeze(-1)
            # update training points
            self.train_x = torch.cat([self.train_x, new_x], dim=0)
            self.train_obj = torch.cat([self.train_obj, new_obj], dim=0)
            self.train_c = torch.cat([self.train_c, new_c], dim=0)

            # If initial size not reached, returns 1 additionnal solution
            if len(self.train_obj) < self.initial_size:
                return self.search_space.random_point(1), {
                    "iteration": self.iterations,
                    "algorithm": "AddInitCArBO",
                    "model": -1,
                }
            else:
                if self.time_budget:
                    elapsed = time.time() - self.start_time
                    alpha = (self.budget - elapsed) / (self.budget)
                else:
                    alpha = (self.budget - self.iterations) / (self.budget)

                train_obj_std = -standardize(self.train_obj)  # standardize objective

                with gpytorch.settings.max_cholesky_size(self.cholesky_size):
                    mll, model = self._initialize_model(
                        self.train_x,
                        train_obj_std,
                        state_dict=None,
                    )
                    try:
                        fit_gpytorch_mll(mll)
                    except ModelFittingError:
                        return self.search_space.random_point(len(Y)), {
                            "iteration": self.iterations,
                            "algorithm": "FailedCArBO",
                            "model": -1,
                        }

                cost_model = CostModelGP(self.train_x, self.train_c)
                self.acqf_kwargs["best_f"] = torch.max(train_obj_std)
                if "X_baseline" in self.acquisition.__init__.__code__.co_varnames:
                    self.acqf_kwargs["X_baseline"] = (self.train_x,)

                # Build acqf kwargs
                acqf = AcquisitionWithCost(
                    acquisition=self.acquisition,
                    model=model,
                    cost_model=cost_model,
                    alpha=alpha,
                    **self.acqf_kwargs,
                )

                # optimize and get new observation
                new_x, acqf_value = self._optimize_acqf_and_get_observation(acqf)

                return new_x.cpu().numpy().tolist(), {
                    "acquisition": acqf_value.cpu().item(),
                    "algorithm": "CArBO",
                    "model": self.iterations,
                }
