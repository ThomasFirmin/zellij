# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)


from __future__ import annotations
from zellij.core.errors import InputError
from zellij.core.metaheuristic import UnitMetaheuristic

from typing import Tuple, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.search_space import UnitSearchspace

import torch
import numpy as np

import gpytorch

from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood

from botorch.utils import standardize
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import ExpectedImprovement
from botorch import fit_gpytorch_model
from botorch.exceptions import ModelFittingError

from torch.quasirandom import SobolEngine

import logging

logger = logging.getLogger("zellij.BO")


class BayesianOptimization(UnitMetaheuristic):
    """BayesianOptimization

    Bayesian optimization (BO) is a surrogate based optimization method which
    interpolates the actual loss function with a surrogate model, here a
    gaussian process.
    It is based on `BoTorch <https://botorch.org/>`_ and `GPyTorch <https://gpytorch.ai/>`__.

    Attributes
    ----------
    search_space : UnitSearchspace
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

    Examples
    --------
    >>> from zellij.core import UnitSearchspace, ArrayVar, FloatVar
    >>> from zellij.utils import FloatMinMax, ArrayDefaultC
    >>> from zellij.core import Experiment, Loss, Minimizer, Calls
    >>> from zellij.strategies.continuous import BayesianOptimization


    >>> @Loss(objective=Minimizer("obj"))
    ... def himmelblau(x):
    ...    res = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    ...    return {"obj": res}


    >>> a = ArrayVar(
    ...     FloatVar("f1", -5, 5, converter=FloatMinMax()),
    ...     FloatVar("i2", -5, 5, converter=FloatMinMax()),
    ...     converter= ArrayDefaultC(),
    ... )
    >>> sp = UnitSearchspace(a)
    >>> opt = BayesianOptimization(sp)
    >>> stop = Calls(himmelblau, 100)
    >>> exp = Experiment(opt, himmelblau, stop)
    >>> exp.run()
    >>> print(f"f({himmelblau.best_point})={himmelblau.best_score}")
    f([-2.8044706937942547, 3.1283592102618414])=0.0003616033800198142
    >>> print(f"Calls: {himmelblau.calls}")
    Calls: 100
    """

    def __init__(
        self,
        search_space: UnitSearchspace,
        verbose: bool = True,
        surrogate=SingleTaskGP,
        mll=ExactMarginalLogLikelihood,
        likelihood=GaussianLikelihood,
        acquisition=ExpectedImprovement,
        initial_size: int = 10,
        gpu: bool = False,
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

        ##############
        # PARAMETERS #
        ##############

        self.acquisition = acquisition
        self.surrogate = surrogate
        self.mll = mll
        self.likelihood = likelihood
        self.initial_size = initial_size

        self.kwargs = kwargs

        #############
        # VARIABLES #
        #############

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
                logger.warning(f"A GPU is not available for BayesianOptimization.")
        else:
            self.device = torch.device("cpu")

        self.dtype = torch.double

        self.sobol = SobolEngine(dimension=self.search_space.size, scramble=True)
        self._build_kwargs()

    def _generate_initial_data(self, n: int) -> torch.Tensor:
        X_init = self.sobol.draw(n=n).to(dtype=self.dtype, device=self.device)
        lower = torch.tensor(
            self.search_space.lower, device=self.device, dtype=self.dtype
        )
        upper = torch.tensor(
            self.search_space.upper, device=self.device, dtype=self.dtype
        )
        X_init = X_init * (upper - lower) + lower
        return X_init

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
            options={"batch_limit": 5, "maxiter": 200},
        )

        return candidates.detach(), acqf_value

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

    def reset(self):
        """reset()

        reset :code:`Bayesian_optimization` to its initial state.

        """
        self.initialized = False
        self.train_x = torch.empty((0, self.search_space.size))
        self.train_obj = torch.empty((0, 1))
        self.state_dict = {}

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
            self.iteration = 1

            # call helper functions to generate initial training data and initialize model
            train_x = self._generate_initial_data(n=self.initial_size)
            self.initialized = True
            return train_x.cpu().numpy().tolist(), {
                "acquisition": 0,
                "algorithm": "InitBO",
            }
        elif X is None or Y is None:
            raise InputError(
                "After initialization Bayesian optimization must receive non-empty X and Y in forward."
            )
        else:
            self.iteration += 1

            npx = np.array(X)
            npy = np.array(Y)
            mask = np.isfinite(npy)

            mx = npx[mask]
            my = npy[mask]

            if len(my) > 0:
                new_x = torch.tensor(mx, dtype=self.dtype)
                new_obj = torch.tensor(my, dtype=self.dtype).unsqueeze(-1)

                # update training points
                self.train_x = torch.cat([self.train_x, new_x])
                self.train_obj = torch.cat([self.train_obj, new_obj])

                train_obj_std = -standardize(self.train_obj)

                # reinitialize the models so they are ready for fitting on next iteration
                # use the current state dict to speed up fitting
                mll, model = self._initialize_model(
                    self.train_x,
                    train_obj_std,
                    self.state_dict,
                )

                self.state_dict = model.state_dict()

                try:
                    with gpytorch.settings.max_cholesky_size(300):
                        # run N_BATCH rounds of BayesOpt after the initial random batch
                        # fit the models
                        fit_gpytorch_model(mll)

                        # Add potentially usefull kwargs for acqf kwargs
                        self.acqf_kwargs["best_f"] = torch.max(train_obj_std)
                        if (
                            "X_baseline"
                            in self.acquisition.__init__.__code__.co_varnames
                        ):
                            self.acqf_kwargs["X_baseline"] = (self.train_x,)

                        # Build acqf kwargs
                        acqf = self.acquisition(model=model, **self.acqf_kwargs)

                        # optimize and get new observation
                        new_x, acqf_value = self._optimize_acqf_and_get_observation(
                            acqf
                        )

                        return new_x.cpu().numpy().tolist(), {
                            "acquisition": acqf_value.cpu().item(),
                            "algorithm": "BO",
                        }
                except ModelFittingError:
                    new_x = self._generate_initial_data(1)
                    return new_x.cpu().numpy().tolist(), {
                        "acquisition": 0,
                        "algorithm": "ModelFittingError",
                    }
            else:
                new_x = self._generate_initial_data(1)
                return new_x.cpu().numpy().tolist(), {
                    "acquisition": 0,
                    "algorithm": "ResampleBO",
                }
