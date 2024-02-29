# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)
from __future__ import annotations
from abc import ABC, abstractmethod
from zellij.core.errors import InputError
from zellij.core.metaheuristic import UnitMetaheuristic
from zellij.strategies.tools.turbo_state import update_c_state

from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.search_space import UnitSearchspace
    from zellij.strategies.tools.turbo_state import CTurboState

import torch
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.exceptions import ModelFittingError
from botorch.models.transforms import Log

from botorch.fit import fit_gpytorch_mll

from zellij.strategies.tools.turbo_state import (
    ConstrainedTSPerUnitPosteriorSampling,
    Temperature,
)

import numpy as np
import time

import os

import logging

logger = logging.getLogger("zellij.cascbo")


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

    def __init__(self, X, Y_cost, device, dtype):
        assert torch.all(Y_cost > 0)

        X.to(device, dtype=dtype)
        Y_cost.to(device, dtype=dtype)

        super().__init__()
        gp = SingleTaskGP(
            train_X=X,
            train_Y=Y_cost,
            outcome_transform=Log(),
        )
        gp.to(device)
        mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
        fit_gpytorch_mll(mll)
        self.gp = gp

    def forward(self, X):
        return torch.exp(self.gp(X).mean)


class CASCBOI(UnitMetaheuristic):
    """Cost Aware Scalable Constrained Bayesian Optimization

    Works in the unit hypercube. :code:`converter` :ref:`addons` are required.

    See `CASCBO <https://botorch.org/tutorials/scalable_constrained_bo>`_.
    It is based on `BoTorch <https://botorch.org/>`_ and `GPyTorch <https://gpytorch.ai/>`__.

    Attributes
    ----------
    search_space : ContinuousSearchspace
        Search space object containing bounds of the search space
    turbo_state : CTurboState
        :code:`CTurboState` object.
    cooling : Temperature
        Used to determine the probability of accepting expensive solutions.
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
    batch_size : int, default=4
        Number of solutions sampled within the surrogate, to return at each iteration.
    n_canditates : int, default=None
        Number of candidates to sample with the surrogate.
    initial_size : int, default=10
        Size of the initial set of solution to draw randomly.
    cholesky_size : int, default=800
        Maximum size for which Lanczos method is used instead of Cholesky decomposition.
    beam : int, default=2000
        Maximum number of solutions that can be stored and used to compute the Gaussian Process.
    gpu: bool, default=True
        Use GPU if available
    kwargs
        Key word arguments linked to the surrogate, mll or likelihood.

    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is
    :ref:`lf` : Describes what a loss function is in Zellij
    :ref:`sp` : Describes what a loss function is in Zellij

    Examples
    --------
    >>> from zellij.core import UnitSearchspace, ArrayVar, FloatVar
    >>> from zellij.utils import ArrayDefaultC, FloatMinMax
    >>> from zellij.core import Experiment, Loss, Minimizer, Calls
    >>> from zellij.strategies.continuous import SCBO
    >>> from zellij.strategies.tools import CTurboState
    >>> import torch
    >>> import numpy as np

    >>> @Loss(objective=Minimizer("obj"), constraint=["c0", "c1"])
    >>> def himmelblau(x):
    ...     res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...     # coordinates must be <0
    ...     return {"obj": res, "c0": x[0], "c1": x[1]}


    >>> a = ArrayVar(
    ...     FloatVar("f1", -5, 5, converter=FloatMinMax()),
    ...     FloatVar("i2", -5, 5, converter=FloatMinMax()),
    ...     converter=ArrayDefaultC(),
    ... )
    >>> sp = UnitSearchspace(a)
    >>> batch_size = 10
    >>> tstate = CTurboState(sp.size, batch_size, torch.ones(2) * torch.inf)

    >>> opt = SCBO(sp, tstate, batch_size)
    >>> stop = Calls(himmelblau, 100)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run()
    >>> bx = himmelblau.best_point
    >>> by = himmelblau.best_score
    >>> cstr = cstr = np.char.mod("%.2f", himmelblau.best_constraint)
    >>> print(f"f({bx})={by:.2f}, s.t. ({'<0, '.join(cstr)}<0)")
    f([-3.781168491476484, -3.286043016729221])=0.00, s.t. (-3.78<0, -3.29<0)
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 100
    """

    def __init__(
        self,
        search_space: UnitSearchspace,
        turbo_state: CTurboState,
        batch_size: int,
        budget: float,
        temperature: Temperature,
        verbose: bool = True,
        surrogate=SingleTaskGP,
        mll=ExactMarginalLogLikelihood,
        likelihood=GaussianLikelihood,
        n_candidates: Optional[int] = None,
        initial_size: int = 10,
        cholesky_size: int = 800,
        beam: int = 2000,
        gpu: bool = False,
        start_shrinking=0,
        time_budget=False,
        fixed_lbound=None,
        lbound_evolv=None,
        fixed_ubound=None,
        ubound_evolv=None,
        **kwargs,
    ):
        """__init__

        Parameters
        ----------
        search_space : UnitSearchspace
            UnitSearchspace :ref:`sp`.
        turbo_state : CTurboState
            :code:`CTurboState` object.
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
        batch_size : int, default=4
            Number of solutions sampled within the surrogate, to return at each iteration.
        n_canditates : int, default=None
            Number of candidates to sample with the surrogate.
        initial_size : int, default=10
            Size of the initial set of solution to draw randomly.
        cholesky_size : int, default=800
            Maximum size for which Lanczos method is used instead of Cholesky decomposition.
        beam : int, default=2000
            Maximum number of solutions that can be stored and used to compute the Gaussian Process.
        gpu: bool, default=True
            Use GPU if available
        kwargs
            Key word arguments linked to the surrogate, mll or likelihood.
        """

        super().__init__(search_space, verbose)

        ##############
        # PARAMETERS #
        ##############
        self.surrogate = surrogate
        self.mll = mll
        self.likelihood = likelihood
        self.batch_size = batch_size
        self.budget = budget
        self.temperature = temperature
        self.time_budget = time_budget

        self.start_shriking = start_shrinking
        self.fixed_lbound = fixed_lbound
        self.fixed_ubound = fixed_ubound
        self.lbound_evolv = lbound_evolv
        self.ubound_evolv = ubound_evolv

        self.start_time = time.time()

        self.n_candidates = n_candidates
        self.initial_size = initial_size

        self.beam = beam

        self.kwargs = kwargs

        #############
        # VARIABLES #
        #############
        self.nconstraint = None

        self.turbo_state = turbo_state

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
        # Prior constraints
        self.train_c = None
        # Cost objective
        self.train_cost = torch.empty((0, 1), dtype=self.dtype, device=self.device)

        self.sobol = SobolEngine(dimension=self.search_space.size, scramble=True)

        self._build_kwargs()

        self.cmodels_list = None

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

    def generate_batch(
        self,
        state: CTurboState,
        model,  # GP model
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        batch_size,
        n_candidates,  # Number of candidates for Thompson sampling
        constraint_model,
        cost_model,
        best_obj,
        best_cost,
        alpha,
        current_temp,
    ):
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()

        # Add weights based trust region
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        ################################

        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        if self.fixed_lbound:
            tr_lb[self.fixed_lbound] = 0.0
            if self.lbound_evolv:
                ratio = self.lbound_evolv(alpha)
                tr_lb[self.fixed_ubound] *= ratio

        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
        if self.fixed_ubound:
            tr_lb[self.fixed_ubound] = 1.0
            if self.ubound_evolv:
                ratio = self.ubound_evolv(alpha)
                tr_lb[self.fixed_ubound] *= ratio

        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=self.dtype, device=self.device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(n_candidates, dim, dtype=self.dtype, device=self.device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=self.device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        constrained_thompson_sampling = ConstrainedTSPerUnitPosteriorSampling(
            model=model,
            constraint_model=constraint_model,
            cost_model=cost_model,
            temperature=current_temp,
            best_score=best_obj,
            best_cost=best_cost,
            replacement=False,
        )

        with torch.no_grad():  # We don't need gradients when using TS
            X_next = constrained_thompson_sampling(X_cand, num_samples=batch_size)

        return X_next.detach()

    def reset(self):
        """reset()

        reset :code:`Bayesian_optimization` to its initial state.

        """
        self.initialized = False
        self.train_x = torch.empty(
            (0, self.search_space.size), dtype=self.dtype, device=self.device
        )
        self.obj = torch.empty((0, 1), dtype=self.dtype, device=self.device)
        self.train_c = None
        self.train_cost = torch.empty((0, 1), dtype=self.dtype, device=self.device)

    def forward(
        self,
        X: Optional[list] = None,
        Y: Optional[np.ndarray] = None,
        secondary: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
    ) -> Tuple[List[list], dict]:
        """forward

        Abstract method describing one step of the :ref:`meta`.

        Parameters
        ----------
        X : list
            List of points.
        Y : numpy.ndarray[float]
            List of loss values.
        secondary : np.ndarray, optional
            :code:`constraint` numpy ndarray of floats. See :ref:`lf` for more info.
        constraint : np.ndarray, optional
            :code:`constraint` numpy ndarray of floats. See :ref:`lf` for more info.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Dictionnary of additionnal information linked to :code:`points`.
        """
        torch.cuda.empty_cache()

        if self.turbo_state.restart_triggered:
            self.initialized = False
            self.turbo_state.reset()

        if not self.initialized:
            # call helper functions to generate initial training data and initialize model
            train_x = self._generate_initial_data()
            self.initialized = True
            return train_x, {
                "iteration": self.iterations,
                "algorithm": "InitCASCBOI",
                "length": 1.0,
                "trestart": self.turbo_state.restart_triggered,
                "model": -1,
                "temperature": -1,
                "beam": len(self.train_obj),
            }
        elif X is None or Y is None or secondary is None or constraint is None:
            raise InputError(
                "After initialization CASCBOI must receive non-empty X, Y, secondary and constraint in forward."
            )
        else:
            if self.train_c is None or self.nconstraint is None:
                self.nconstraint = constraint.shape[1]
                # Prior constraints
                self.train_c = torch.empty(
                    (0, self.nconstraint),
                    dtype=self.dtype,
                    device=self.device,
                )
                self.cmodels_list = [None] * self.nconstraint

            torch.cuda.empty_cache()
            self.iterations += 1

            new_x = torch.tensor(X, dtype=self.dtype, device=self.device)
            new_obj = -torch.tensor(Y, dtype=self.dtype, device=self.device).unsqueeze(
                -1
            )
            new_c = torch.tensor(constraint, dtype=self.dtype, device=self.device)

            new_cost = torch.tensor(
                secondary[:, 0], dtype=self.dtype, device=self.device
            ).unsqueeze(-1)

            # update training points
            self.train_x = torch.cat([self.train_x, new_x], dim=0)
            self.train_obj = torch.cat([self.train_obj, new_obj], dim=0)
            self.train_c = torch.cat([self.train_c, new_c], dim=0)
            self.train_cost = torch.cat([self.train_cost, new_cost], dim=0)

            # Remove worst solutions from the beam
            if len(self.train_x) > self.beam:
                sidx = torch.argsort(self.train_obj.squeeze(), descending=True)

                self.train_x = self.train_x[sidx]
                self.train_obj = self.train_obj[sidx]
                self.train_c = self.train_c[sidx]
                self.train_cost = self.train_cost[sidx]

                violation = self.train_c.sum(dim=1)
                nvidx = violation < 0

                new_x = self.train_x[nvidx][: self.beam]
                new_obj = self.train_obj[nvidx][: self.beam]
                new_c = self.train_c[nvidx][: self.beam]
                new_cost = self.train_cost[nvidx][: self.beam]

                if len(new_x) < self.beam:
                    nfill = self.beam - len(new_x)
                    violated = torch.logical_not(nvidx)
                    v_x = self.train_x[violated]
                    v_obj = self.train_obj[violated]
                    v_c = self.train_c[violated]
                    v_cost = self.train_cost[violated]

                    scidx = torch.argsort(violation[violated].squeeze())[:nfill]

                    # update training points
                    self.train_x = torch.cat([new_x, v_x[scidx]], dim=0)
                    self.train_obj = torch.cat([new_obj, v_obj[scidx]], dim=0)
                    self.train_c = torch.cat([new_c, v_c[scidx]], dim=0)
                    self.train_cost = torch.cat([new_cost, v_cost[scidx]], dim=0)

                else:
                    self.train_x = new_x
                    self.train_obj = new_obj
                    self.train_c = new_c
                    self.train_cost = new_cost

            # If initial size not reached, returns 1 additionnal solution
            if len(self.train_obj) < self.initial_size:
                return self.search_space.random_point(1), {
                    "iteration": self.iterations,
                    "algorithm": "AddInitCASCBOI",
                    "length": 1.0,
                    "trestart": self.turbo_state.restart_triggered,
                    "model": -1,
                    "temperature": -1,
                    "beam": len(self.train_obj),
                }
            else:
                # Compute temperature
                if self.time_budget:
                    elapsed = time.time() - self.start_time
                else:
                    elapsed = self.iterations

                alpha = elapsed / self.budget
                current_temp = 1 - np.clip(self.temperature.temperature(alpha), 0, 1)

                if alpha > self.start_shriking:
                    self.turbo_state = update_c_state(
                        state=self.turbo_state, Y_next=new_obj, C_next=new_c
                    )

                with gpytorch.settings.max_cholesky_size(self.cholesky_size):
                    # reinitialize the models so they are ready for fitting on next iteration
                    # use the current state dict to speed up fitting
                    mll, model = self._initialize_model(
                        self.train_x,
                        self.train_obj,
                        state_dict=None,
                    )

                    try:
                        fit_gpytorch_mll(mll)
                    except ModelFittingError:
                        return self.search_space.random_point(len(Y)), {
                            "iteration": self.iterations,
                            "algorithm": "FailedCASCBOI",
                            "length": 1.0,
                            "trestart": self.turbo_state.restart_triggered,
                            "model": -1,
                            "temperature": current_temp,
                            "beam": len(self.train_obj),
                        }

                    # Update constraint models
                    for i in range(self.nconstraint):
                        cmll, cmodel = self._initialize_model(
                            self.train_x,
                            self.train_c[:, i].unsqueeze(-1),
                            state_dict=None,
                        )
                        try:
                            fit_gpytorch_mll(cmll)
                            self.cmodels_list[i] = cmodel  # type: ignore

                        except ModelFittingError:
                            logger.warning(
                                f"In SCBO, ModelFittingError for constraint{i}, previous fitted model will be used."
                            )

                    # Update Cost model
                    cost_model = CostModelGP(
                        self.train_x, self.train_cost, self.device, self.dtype
                    )

                    # optimize and get new observation
                    violation = self.train_c.sum(dim=1)
                    nvidx = violation < 0
                    if torch.any(nvidx):
                        print("SOME NON VIOLATED")
                        best_arg = torch.argmax(self.train_obj[nvidx])
                        best_obj = self.train_obj[nvidx][best_arg].item()
                        best_cost = self.train_cost[nvidx][best_arg].item()
                    else:
                        print("ALL VIOLATED")
                        best_arg = torch.argmin(violation)
                        best_obj = self.train_obj[best_arg].item()
                        best_cost = self.train_cost[best_arg].item()

                    print(f"BEST COST : {best_cost}")

                    # Compute temperature
                    if self.time_budget:
                        elapsed = time.time() - self.start_time
                    else:
                        elapsed = self.iterations

                    alpha = elapsed / self.budget
                    current_temp = 1 - np.clip(
                        self.temperature.temperature(alpha), 0, 1
                    )

                    new_x = self.generate_batch(
                        state=self.turbo_state,
                        model=model,
                        X=self.train_x,
                        Y=self.train_obj,
                        batch_size=self.batch_size,
                        n_candidates=self.n_candidates,
                        constraint_model=ModelListGP(*self.cmodels_list),
                        cost_model=cost_model,
                        best_obj=best_obj,
                        best_cost=best_cost,
                        alpha=alpha,
                        current_temp=current_temp,
                    )

                    if self._save:
                        self.save(model, self.cmodels_list)

                    return new_x.cpu().numpy().tolist(), {
                        "iteration": self.iterations,
                        "algorithm": "CASCBOI",
                        "length": self.turbo_state.length,
                        "trestart": self.turbo_state.restart_triggered,
                        "model": self.models_number,
                        "temperature": current_temp,
                        "beam": len(self.train_obj),
                    }

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["cmodels_list"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save(self, model, cmodels):
        path = self._save
        foldername = os.path.join(path, "cascboi")
        if not os.path.exists(foldername):
            os.makedirs(foldername)

        std_dict = model.state_dict()
        std_dict["nlengthscale"] = model.covar_module.base_kernel.lengthscale

        torch.save(
            std_dict,
            os.path.join(foldername, f"{self.models_number}_model.pth"),
        )
        for idx, m in enumerate(cmodels):
            std_dict = m.state_dict()
            std_dict["nlengthscale"] = m.covar_module.base_kernel.lengthscale
            torch.save(
                std_dict,
                os.path.join(foldername, f"{self.models_number}_c{idx}model.pth"),
            )
        self.models_number += 1
