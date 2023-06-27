# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-23T14:51:22+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.metaheuristic import ContinuousMetaheuristic

import torch
import numpy as np
from botorch.models import SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.generation import MaxPosteriorSampling

from botorch import fit_gpytorch_model
from botorch.fit import fit_gpytorch_mll

import math
from dataclasses import dataclass

from torch.quasirandom import SobolEngine
import gpytorch

import logging

logger = logging.getLogger("zellij.BO")


class Bayesian_optimization(ContinuousMetaheuristic):
    """Bayesian_optimization

    Bayesian optimization (BO) is a surrogate based optimization method which
    interpolates the actual loss function with a surrogate model, here it is a
    gaussian process. By sampling into this surrogate,
    BO determines promising points, which are worth to evaluate with the actual
    loss function. Once done, the gaussian process is updated using results
    obtained by evaluating these promising solutions with the loss function.

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

    Examples
    --------
    >>> from zellij.core import Loss, Threshold, Experiment
    >>> from zellij.core import ContinuousSearchspace, FloatVar, ArrayVar
    >>> from zellij.utils.benchmarks import himmelblau
    >>> from zellij.strategies import Bayesian_optimization
    >>> import botorch
    >>> import gpytorch
    ...
    >>> lf = Loss()(himmelblau)
    >>> sp = ContinuousSearchspace(ArrayVar(FloatVar("a",-5,5), FloatVar("b",-5,5)),lf)
    >>> stop = Threshold(lf, 'calls', 100)
    >>> bo = Bayesian_optimization(sp,
    ...       acquisition=botorch.acquisition.monte_carlo.qExpectedImprovement,
    ...       q=5)
    >>> exp = Experiment(bo, stop)
    >>> exp.run()
    >>> print(f"Best solution:f({lf.best_point})={lf.best_score}")
    """

    def __init__(
        self,
        search_space,
        verbose=True,
        surrogate=SingleTaskGP,
        mll=ExactMarginalLogLikelihood,
        likelihood=GaussianLikelihood,
        acquisition=ExpectedImprovement,
        initial_size=10,
        gpu=False,
        **kwargs,
    ):
        """Short summary.

        Parameters
        ----------
        search_space : Searchspace
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
        **kwargs
            Key word arguments linked to the surrogate and the acquisition function.
        """

        super().__init__(search_space, verbose)

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
        self.state_dict = None

        # Determine if BO is initialized or not
        self.initialized = False

        # Number of iterations
        self.iterations = 0

        if gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        self.dtype = torch.double

        self.sobol = SobolEngine(dimension=self.search_space.size, scramble=True)
        self._build_kwargs()

    def _generate_initial_data(self, n):
        X_init = self.sobol.draw(n=n).to(dtype=self.dtype, device=self.device)
        lower = torch.tensor(
            self.search_space.lower, device=self.device, dtype=self.dtype
        )
        upper = torch.tensor(
            self.search_space.upper, device=self.device, dtype=self.dtype
        )
        X_init = X_init * (upper - lower) + lower
        return X_init

    def _initialize_model(self, train_x, train_obj, state_dict=None):
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
        if state_dict is not None:
            model.load_state_dict(state_dict)

        model.to(self.device)

        return mll, model

    def _optimize_acqf_and_get_observation(self, acq_func, restarts=10, raw=512):
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
        self.model_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.surrogate.__init__.__code__.co_varnames
        }
        self.likelihood_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.likelihood.__init__.__code__.co_varnames
        }
        self.mll_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.mll.__init__.__code__.co_varnames
        }
        self.acqf_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.acquisition.__init__.__code__.co_varnames
        }
        print(
            self.model_kwargs, self.likelihood_kwargs, self.mll_kwargs, self.acqf_kwargs
        )

    def reset(self):
        """reset()

        reset :code:`Bayesian_optimization` to its initial state.

        """
        self.initialized = False
        self.train_x = torch.empty((0, self.search_space.size))
        self.train_y = torch.empty((0, 1))
        self.state_dict = None

    def forward(self, X, Y, constraint=None):
        """forward

        Runs one step of BO.

        Parameters
        ----------
        X : list
            List of previously computed points
        Y : list
            List of loss value linked to :code:`X`.
            :code:`X` and :code:`Y` must have the same length.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """

        if not self.initialized:
            self.iteration = 1

            # call helper functions to generate initial training data and initialize model
            train_x = self._generate_initial_data(n=self.initial_size)
            self.initialized = True
            return train_x.cpu().numpy(), {"acquisition": 0, "algorithm": "InitBO"}
        else:
            if (X is not None and Y is not None) and (len(X) > 0 and len(Y) > 0):
                self.iteration += 1

                mask = np.isfinite(Y)
                X = np.array(X)
                Y = -np.array(Y)  # negate Y, BoTorch -> max, Zellij -> min

                mx = X[mask]
                my = Y[mask]

                if len(my) > 0:
                    new_x = torch.tensor(mx, dtype=self.dtype)
                    new_obj = torch.tensor(my, dtype=self.dtype).unsqueeze(-1)

                    # update training points
                    self.train_x = torch.cat([self.train_x, new_x])
                    self.train_obj = torch.cat([self.train_obj, new_obj])

                    # reinitialize the models so they are ready for fitting on next iteration
                    # use the current state dict to speed up fitting
                    mll, model = self._initialize_model(
                        self.train_x,
                        self.train_obj,
                        self.state_dict,
                    )

                    self.state_dict = model.state_dict()

                    try:
                        with gpytorch.settings.max_cholesky_size(float("inf")):
                            # run N_BATCH rounds of BayesOpt after the initial random batch
                            # fit the models
                            fit_gpytorch_model(mll)

                            # Add potentially usefull kwargs for acqf kwargs
                            self.acqf_kwargs[
                                "best_f"
                            ] = self.search_space.loss.best_score
                            self.acqf_kwargs["X_baseline"] = (self.train_x,)

                            # Build acqf kwargs
                            acqf = self.acquisition(model=model, **self.acqf_kwargs)

                            # optimize and get new observation
                            new_x, acqf_value = self._optimize_acqf_and_get_observation(
                                acqf
                            )

                            return new_x.cpu().numpy(), {
                                "acquisition": acqf_value.cpu().item(),
                                "algorithm": "BO",
                            }
                    except:
                        new_x = self._generate_initial_data(1)
                        return new_x.cpu().numpy(), {
                            "acquisition": 0,
                            "algorithm": "ResampleBO",
                        }
                else:
                    new_x = self._generate_initial_data(1)
                    return new_x.cpu().numpy(), {
                        "acquisition": 0,
                        "algorithm": "ResampleBO",
                    }


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # type: ignore
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True

    return state


class TuRBO(ContinuousMetaheuristic):
    """Trust region Bayesian Optimization

    /!\ Works in the unit hypercube. :code:`converter` :ref:`addons` are required.

    See `TuRBO <https://botorch.org/tutorials/turbo_1>`_.
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

    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is
    :ref:`lf` : Describes what a loss function is in Zellij
    :ref:`sp` : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        search_space,
        verbose=True,
        surrogate=SingleTaskGP,
        mll=ExactMarginalLogLikelihood,
        likelihood=GaussianLikelihood,
        batch_size=4,
        n_candidates=None,
        initial_size=10,
        gpu=False,
        **kwargs,
    ):
        """Short summary.

        Parameters
        ----------
        search_space : Searchspace
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
        **kwargs
            Key word arguments linked to the surrogate and the acquisition function.
        """

        super().__init__(search_space, verbose)

        ##############
        # PARAMETERS #
        ##############

        self.surrogate = surrogate
        self.mll = mll
        self.likelihood = likelihood

        self.batch_size = batch_size
        self.n_candidates = n_candidates
        self.initial_size = initial_size

        self.turbo_state = TurboState(self.search_space.size, batch_size=batch_size)

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
        self.train_obj = torch.empty((0, 1), dtype=self.dtype, device=self.device)

        self.sobol = SobolEngine(dimension=self.search_space.size, scramble=True)

        self._build_kwargs()

    def _build_kwargs(self):
        self.model_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.surrogate.__init__.__code__.co_varnames
        }
        self.likelihood_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.likelihood.__init__.__code__.co_varnames
        }
        self.mll_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.mll.__init__.__code__.co_varnames
        }
        print(self.model_kwargs, self.likelihood_kwargs, self.mll_kwargs)

    def _generate_initial_data(self):
        X_init = self.sobol.draw(n=self.initial_size).to(
            dtype=self.dtype, device=self.device
        )
        return X_init.detach()

    def _initialize_model(self, train_x, train_obj, state_dict=None):
        train_x.to(self.device, dtype=self.dtype)
        train_obj.to(self.device, dtype=self.dtype)

        train_Y = (train_obj - train_obj.mean()) / train_obj.std()
        likelihood = self.likelihood(**self.likelihood_kwargs)

        # define models for objective and constraint
        model = self.surrogate(
            train_x,
            train_Y,
            likelihood=likelihood,
            **self.model_kwargs,
        )

        mll = self.mll(
            model.likelihood,
            model,
            **self.mll_kwargs,
        )

        # load state dict if it is passed
        if state_dict is not None:
            model.load_state_dict(state_dict)

        model.to(self.device)

        return mll, model, train_Y

    def generate_batch(
        self,
        state,
        model,  # GP model
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        batch_size,
        n_candidates,  # Number of candidates for Thompson sampling
    ):
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

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
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

        return X_next.detach()

    def reset(self):
        """reset()

        reset :code:`Bayesian_optimization` to its initial state.

        """
        self.initialized = False
        self.train_x = torch.empty(
            (0, self.search_space.size), dtype=self.dtype, device=self.device
        )
        self.train_y = torch.empty((0, 1), dtype=self.dtype, device=self.device)

    def forward(self, X, Y, constraint=None):
        """forward

        Runs one step of BO.

        Parameters
        ----------
        X : list
            List of previously computed points
        Y : list
            List of loss value linked to :code:`X`.
            :code:`X` and :code:`Y` must have the same length.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """

        if not self.initialized:
            self.iteration = 1

            # call helper functions to generate initial training data and initialize model
            train_x = self._generate_initial_data()
            self.initialized = True
            return train_x.cpu().numpy(), {
                "iteration": self.iteration,
                "algorithm": "InitTuRBO",
            }
        else:
            if (X and Y) and (len(X) > 0 and len(Y) > 0):
                self.iteration += 1

                mask = np.isfinite(Y)
                X = np.array(X)
                Y = np.array(Y)  # negate Y, BoTorch -> max, Zellij -> min

                mx = X[mask]
                my = Y[mask]

                if len(my) > 0:
                    new_x = torch.tensor(mx, dtype=self.dtype, device=self.device)
                    new_obj = torch.tensor(
                        my, dtype=self.dtype, device=self.device
                    ).unsqueeze(-1)

                    self.turbo_state = update_state(
                        state=self.turbo_state, Y_next=new_obj
                    )

                    # update training points
                    self.train_x = torch.cat([self.train_x, new_x])
                    self.train_obj = torch.cat([self.train_obj, new_obj])

                    # reinitialize the models so they are ready for fitting on next iteration
                    # use the current state dict to speed up fitting
                    mll, model, train_Y = self._initialize_model(
                        self.train_x,
                        self.train_obj,
                        state_dict=None,
                    )
                    with gpytorch.settings.max_cholesky_size(float("inf")):
                        try:
                            # run N_BATCH rounds of BayesOpt after the initial random batch
                            # fit the models
                            fit_gpytorch_mll(mll)

                            # optimize and get new observation
                            new_x = self.generate_batch(
                                state=self.turbo_state,
                                model=model,
                                X=self.train_x,
                                Y=train_Y,
                                batch_size=self.batch_size,
                                n_candidates=self.n_candidates,
                            )
                            return new_x.cpu().numpy(), {
                                "iteration": self.iteration,
                                "algorithm": "TuRBO",
                            }
                        except:
                            return self.sobol.draw(n=len(Y)).cpu().numpy(), {
                                "iteration": self.iteration,
                                "algorithm": "FailedTuRBO",
                            }
                else:
                    return self.sobol.draw(n=len(Y)).cpu().numpy(), {
                        "iteration": self.iteration,
                        "algorithm": "ResampleTuRBO",
                    }
            else:
                return self.sobol.draw(n=1).cpu().numpy(), {
                    "iteration": self.iteration,
                    "algorithm": "ResampleTuRBO",
                }
