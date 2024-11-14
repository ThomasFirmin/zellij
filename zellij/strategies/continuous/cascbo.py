# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)
from __future__ import annotations
from abc import ABC, abstractmethod
from zellij.core.errors import InputError
from zellij.core.metaheuristic import (
    UnitMetaheuristic,
    MultiObjective,
    Constrained,
)
from zellij.strategies.tools.turbo_state import (
    iupdate_c_state,
    iget_best_index_for_batch,
)

from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.search_space import UnitSearchspace
    from zellij.strategies.tools.turbo_state import ICTurboState

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

from copy import deepcopy

from zellij.strategies.tools.turbo_state import (
    ConstrainedTSPerUnitPosteriorSampling,
    Temperature,
)

import numpy as np
import time
from datetime import datetime

import os
import gc


import logging

logger = logging.getLogger("zellij.cascbo")


class CASCBO(UnitMetaheuristic, MultiObjective, Constrained):
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
        turbo_state: ICTurboState,
        batch_size: int,
        budget: float,
        temperature: Temperature,
        covar_module,
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

        ####################
        # INFO TO RETRIEVE # # See metaheuristic
        ####################
        self.info = ["length", "successes", "failures"]

        ##############
        # PARAMETERS #
        ##############
        self.surrogate = surrogate
        self.covar_module = covar_module
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

        if isinstance(gpu, str):
            self.device = torch.device(gpu)
        elif gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

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

    def _generate_initial_data(self, n, alpha) -> List[list]:

        x = self.search_space.random_point(n)
        if self.lbound_evolv or self.ubound_evolv:
            x = np.array(x)
            lbounds = np.zeros(self.search_space.size)
            ubounds = np.ones(self.search_space.size)

            if self.lbound_evolv:
                minb = self.lbound_evolv(alpha)
                lbounds[self.fixed_lbound] = minb

            if self.ubound_evolv:
                maxb = self.ubound_evolv(alpha)
                ubounds[self.fixed_ubound] = maxb

            x = (x * (ubounds - lbounds) + lbounds).tolist()

        return x

    # Initialize a surrogate
    def _initialize_model(
        self,
        train_x: torch.Tensor,
        train_obj: torch.Tensor,
        state_dict: Optional[dict] = None,
    ):
        train_x = train_x.to(self.device, dtype=self.dtype)
        train_obj = train_obj.to(self.device, dtype=self.dtype)

        likelihood = self.likelihood(**self.likelihood_kwargs)
        covar_module = deepcopy(self.covar_module)

        # define models for objective and constraint
        model = self.surrogate(
            train_x,
            train_obj,
            covar_module=covar_module,
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

    # Initialize a surrogate
    def _initialize_cost_model(
        self,
        train_x: torch.Tensor,
        train_obj: torch.Tensor,
        state_dict: Optional[dict] = None,
    ):
        train_x = train_x.to(self.device, dtype=self.dtype)
        train_obj = train_obj.to(self.device, dtype=self.dtype)

        likelihood = self.likelihood(**self.likelihood_kwargs)
        covar_module = deepcopy(self.covar_module)

        # define models for objective and constraint
        model = self.surrogate(
            train_x,
            train_obj,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Log(),
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
        state: ICTurboState,
        model,  # GP model
        constraint_model,
        cost_model,
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        C,  # Function Constraints
        Cost,  # Function Costs
        batch_size,
        n_candidates,  # Number of candidates for Thompson sampling
        alpha,
        current_temp,
    ):
        assert X.min() >= 0.0 and X.max() <= 1.0
        assert torch.all(torch.isfinite(Y))
        assert torch.all(Cost > 0)

        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = state.best_point.clone().to(dtype=self.dtype)
        best_obj = state.best_value

        # Add weights based trust region
        # weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        # weights = weights / weights.mean()
        # weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        ################################

        tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
        if self.fixed_lbound:
            tr_lb[self.fixed_lbound] = 0.0
            if self.lbound_evolv:
                tr_lb[self.fixed_lbound] = self.lbound_evolv(alpha)

        self.turbo_state.current_lbounds = tr_lb

        tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)
        if self.fixed_ubound:
            tr_ub[self.fixed_ubound] = 1.0
            if self.ubound_evolv:
                tr_ub[self.fixed_ubound] = self.ubound_evolv(alpha)

        self.turbo_state.current_ubounds = tr_ub

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
            best_score=best_obj,
            temperature=current_temp,
            replacement=False,
            device=self.device,
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

    def prune(self, X, Y, constraints, cost):
        # Remove worst solutions from the beam
        sidx = torch.argsort(Y.squeeze(), descending=True)

        X = X[sidx]
        Y = Y[sidx]
        constraints = constraints[sidx]
        cost = cost[sidx]

        violation = constraints.sum(dim=1)
        nvidx = violation < 0

        new_x = X[nvidx][: self.beam]
        new_obj = Y[nvidx][: self.beam]
        new_c = constraints[nvidx][: self.beam]
        new_cost = cost[nvidx][: self.beam]

        if len(new_x) < self.beam:
            nfill = self.beam - len(new_x)
            violated = torch.logical_not(nvidx)
            v_x = X[violated]
            v_obj = Y[violated]
            v_c = constraints[violated]
            v_cost = cost[violated]

            scidx = torch.argsort(violation[violated].squeeze())[:nfill]

            # update training points
            filled_x = torch.cat([new_x, v_x[scidx]], dim=0)[: self.beam]
            filled_obj = torch.cat([new_obj, v_obj[scidx]], dim=0)[: self.beam]
            filled_c = torch.cat([new_c, v_c[scidx]], dim=0)[: self.beam]
            filled_cost = torch.cat([new_cost, v_cost[scidx]], dim=0)[: self.beam]

        else:
            filled_x = new_x[: self.beam]
            filled_obj = new_obj[: self.beam]
            filled_c = new_c[: self.beam]
            filled_cost = new_cost[: self.beam]

        return filled_x, filled_obj, filled_c, filled_cost

    def forward(
        self,
        X: Optional[list] = None,
        Y: Optional[np.ndarray] = None,
        constraint: Optional[np.ndarray] = None,
        info: Optional[np.ndarray] = None,
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
        constraint : np.ndarray, optional
            :code:`constraint` numpy ndarray of floats. See :ref:`lf` for more info.
        info : np.ndarray, optional
            :code:`info` numpy ndarray of floats. Mandatory information from the :ref:`meta` and linked to the solution.
            Used oly if :ref:`meta`, requires specific informations that were linked to a solution during a previous :code:`forward`.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Dictionnary of additionnal information linked to :code:`points`.
        """

        if Y is not None:
            secondary = Y[:, 1]
            Y = Y[:, 0]

        gc.collect()
        torch.cuda.empty_cache()
        ask_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

        if self.turbo_state.restart_triggered:
            # self.initialized = False
            self.turbo_state.reset()

        if not self.initialized:
            # call helper functions to generate initial training data and initialize model
            train_x = self._generate_initial_data(self.initial_size, 0.0)
            self.initialized = True
            rdict = {
                "iteration": self.iterations,
                "algorithm": "InitCASCBOI",
                "ask_date": ask_date,
                "send_date": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
                "length": 1.0,
                "successes": 0.0,
                "failures": 0.0,
                "trestart": self.turbo_state.restart_triggered,
                "greedy": self.turbo_state.greedy_move,
                "best_value": self.turbo_state.best_value,
                "best_cost": self.turbo_state.best_cost,
            }
            for idx, c in enumerate(self.turbo_state.best_constraint_values):
                rdict[f"best_c{idx}"] = c.cpu().item()
            rdict["model"] = -1
            rdict["temperature"] = -1
            rdict["beam"] = len(self.train_obj)
            return train_x, rdict, {}
        elif (
            X is None
            or Y is None
            or secondary is None
            or constraint is None
            or info is None
        ):
            raise InputError(
                "After initialization CASCBOI must receive non-empty X, Y, secondary, constraint and info in forward."
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

            self.iterations += 1

            new_x = torch.tensor(X, dtype=self.dtype, device=self.device)
            new_obj = -torch.tensor(Y, dtype=self.dtype, device=self.device).unsqueeze(
                -1
            )
            new_c = torch.tensor(constraint, dtype=self.dtype, device=self.device)

            new_cost = torch.tensor(
                secondary, dtype=self.dtype, device=self.device
            ).unsqueeze(-1)

            new_lengths = torch.tensor(
                info[:, 0], dtype=self.dtype, device=self.device
            ).unsqueeze(-1)
            new_successes = torch.tensor(
                info[:, 1], dtype=self.dtype, device=self.device
            ).unsqueeze(-1)
            new_failures = torch.tensor(
                info[:, 2], dtype=self.dtype, device=self.device
            ).unsqueeze(-1)

            # update training points
            self.train_x = torch.cat([self.train_x, new_x], dim=0)
            self.train_obj = torch.cat([self.train_obj, new_obj], dim=0)
            self.train_c = torch.cat([self.train_c, new_c], dim=0)
            self.train_cost = torch.cat([self.train_cost, new_cost], dim=0)

            if len(self.train_x) > self.beam:
                self.train_x, self.train_obj, self.train_c, self.train_cost = (
                    self.prune(
                        self.train_x, self.train_obj, self.train_c, self.train_cost
                    )
                )

            # If initial size not reached, returns 1 additionnal solution
            if len(self.train_obj) < self.initial_size:
                rdict = {
                    "iteration": self.iterations,
                    "algorithm": "AddInitCASCBOI",
                    "ask_date": ask_date,
                    "send_date": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
                    "length": 1.0,
                    "successes": 0.0,
                    "failures": 0.0,
                    "trestart": self.turbo_state.restart_triggered,
                    "greedy": self.turbo_state.greedy_move,
                    "best_value": self.turbo_state.best_value,
                    "best_cost": self.turbo_state.best_cost,
                }
                for idx, c in enumerate(self.turbo_state.best_constraint_values):
                    rdict[f"best_c{idx}"] = c.cpu().item()
                rdict["model"] = -1
                rdict["temperature"] = -1
                rdict["beam"] = len(self.train_obj)
                return self._generate_initial_data(1, 0.0), rdict, {}
            else:
                # Compute temperature
                if self.time_budget:
                    elapsed = time.time() - self.start_time
                else:
                    elapsed = self.iterations

                alpha = elapsed / self.budget
                current_temp = 1 - np.clip(self.temperature.temperature(alpha), 0, 1)

                if alpha > self.start_shriking:
                    self.turbo_state = iupdate_c_state(
                        state=self.turbo_state,
                        X_next=new_x,
                        Y_next=new_obj,
                        C_next=new_c,
                        Cost_next=new_cost,
                        lengths=new_lengths,
                        successes=new_successes,
                        failures=new_failures,
                    )

                with gpytorch.settings.max_cholesky_size(self.cholesky_size):
                    # reinitialize the models so they are ready for fitting on next iteration
                    # use the current state dict to speed up fitting

                    gc.collect()
                    torch.cuda.empty_cache()

                    mll, model = self._initialize_model(
                        self.train_x,
                        self.train_obj,
                        state_dict=None,
                    )

                    try:
                        fit_gpytorch_mll(mll)
                        gc.collect()
                        torch.cuda.empty_cache()
                    except ModelFittingError:

                        rdict = {
                            "iteration": self.iterations,
                            "algorithm": "FailedCASCBOI",
                            "ask_date": ask_date,
                            "send_date": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
                            "length": 1.0,
                            "successes": 0.0,
                            "failures": 0.0,
                            "trestart": self.turbo_state.restart_triggered,
                            "greedy": self.turbo_state.greedy_move,
                            "best_value": self.turbo_state.best_value,
                            "best_cost": self.turbo_state.best_cost,
                        }
                        for idx, c in enumerate(
                            self.turbo_state.best_constraint_values
                        ):
                            rdict[f"best_c{idx}"] = c.cpu().item()
                        rdict["model"] = -1
                        rdict["temperature"] = current_temp
                        rdict["beam"] = len(self.train_obj)

                        return self._generate_initial_data(len(Y), 0.0), rdict, {}
                    model.to("cpu")
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Update constraint models
                    for icon in range(self.nconstraint):
                        gc.collect()
                        torch.cuda.empty_cache()
                        cmll, cmodel = self._initialize_model(
                            self.train_x,
                            self.train_c[:, icon].unsqueeze(-1),
                            state_dict=None,
                        )
                        try:
                            fit_gpytorch_mll(cmll)
                            cmodel.to("cpu")
                            gc.collect()
                            torch.cuda.empty_cache()
                            self.cmodels_list[icon] = cmodel  # type: ignore
                        except ModelFittingError:
                            logger.warning(
                                f"In CASCBOI, ModelFittingError for constraint{icon}, previous fitted model will be used."
                            )
                    # Update Cost model
                    gc.collect()
                    torch.cuda.empty_cache()
                    cost_mll, cost_model = self._initialize_cost_model(
                        self.train_x,
                        self.train_cost,
                        state_dict=None,
                    )
                    try:
                        fit_gpytorch_mll(cost_mll)
                        cost_model.to("cpu")
                        self.cost_model = cost_model
                    except ModelFittingError:
                        logger.warning(
                            f"In CASCBOI, ModelFittingError for Cost, previous fitted model will be used."
                        )
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Compute temperature
                    if self.time_budget:
                        elapsed = time.time() - self.start_time
                    else:
                        elapsed = self.iterations

                    alpha = elapsed / self.budget
                    current_temp = 1 - np.clip(
                        self.temperature.temperature(alpha), 0, 1
                    )

                    gc.collect()
                    torch.cuda.empty_cache()
                    new_x = self.generate_batch(
                        state=self.turbo_state,
                        model=model,
                        constraint_model=ModelListGP(*self.cmodels_list),
                        cost_model=self.cost_model,
                        X=self.train_x,
                        Y=self.train_obj,
                        C=self.train_c,
                        Cost=self.train_cost,
                        batch_size=self.batch_size,
                        n_candidates=self.n_candidates,
                        alpha=alpha,
                        current_temp=current_temp,
                    )

                    if self._save:
                        self.save(model, self.cmodels_list)

                    rdict = {
                        "iteration": self.iterations,
                        "algorithm": "CASCBOI",
                        "ask_date": ask_date,
                        "send_date": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
                        "length": self.turbo_state.length,
                        "successes": self.turbo_state.success_counter,
                        "failures": self.turbo_state.failure_counter,
                        "trestart": self.turbo_state.restart_triggered,
                        "greedy": self.turbo_state.greedy_move,
                        "best_value": self.turbo_state.best_value,
                        "best_cost": self.turbo_state.best_cost,
                    }
                    for idx, c in enumerate(self.turbo_state.best_constraint_values):
                        rdict[f"best_c{idx}"] = c.cpu().item()
                    rdict["model"] = self.models_number
                    rdict["temperature"] = current_temp
                    rdict["beam"] = len(self.train_obj)

                    return new_x.cpu().numpy().tolist(), rdict, {}

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
