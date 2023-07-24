# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-23T14:51:22+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

from typing import Any, Callable, Dict, List, Optional, Union

from zellij.core.metaheuristic import ContinuousMetaheuristic

import torch
import numpy as np
from botorch.models import SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from botorch.generation import MaxPosteriorSampling
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models.transforms.outcome import Standardize
from botorch.models.model_list_gp_regression import ModelListGP

from botorch.fit import fit_gpytorch_mll

from botorch.exceptions import ModelFittingError

import math
from dataclasses import dataclass

from torch.quasirandom import SobolEngine
from torch import Tensor
import gpytorch

import logging

logger = logging.getLogger("zellij.BO")


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


@dataclass
class CTurboState:
    dim: int
    batch_size: int
    best_constraint_values: Tensor
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


@dataclass
class MorboState:
    r"""Hyperparameters for TuRBO.

    Args:
        length_init: Initial edge length for the trust region
        length_min: Minimum edge length
        length_max: Maximum edge length
        success_streak: Number of consecutive successes necessary to increase length
        failure_streak: Number of consecutive failures necessary to decrease length
        n_trust_regions: Total number of trust regions. This is used in failure
            accounting.
        batch_size: Batch size
        eps: The minimum percent improvement in objective that qualifying as a
            "success".
        use_ard: Whether to use ARD when fitting GPs for this trust region.
        trim_trace: A boolean indicating whether to use all data from the trust
            region's trace for model fitting.
        verbose: A boolean indicating whether to print verbose output
        max_tr_size: The maximum number of points in a trust region. This can be
            used to avoid memory issues.
        min_tr_size: The minimum number of points allowed in the trust region. If there
            are too few points in the TR, sobol samples will be used to refill the TR.
        qmc: Whether to use qmc when possible or not
        sample_subset_d: Whether to perturb subset of the dimensions for generating
            discrete X
        track_history: If true, uses the historically observed points and points
            from other trs when the trust regions moves
        fixed_scalarization: If set, a fixed scalarization weight would be used
        max_cholesky_size: Maximum number of training points for which we will use a
            Cholesky decomposition.
        raw_samples: number of discrete points for Thompson Sampling
        n_initial_points: Number of initial sobol points
        n_restart_points: Number of sobol points to evaluate when we restart a TR if
            `init_with_add_sobol=True`
        max_reference_point: The maximum reference point (i.e. this is the closest that
            the reference point can get to the pareto front)
        hypervolume: Whether to use a hypervolume objective for MOO
        winsor_pct: Percentage of worst outcome observations to winsorize
        trun_normal_perturb: Whether to generate discrete points for Thompson sampling
            by perturbing with samples from a zero-mean truncated Gaussian.
        decay_restart_length_alpha: Factor controlling how much to decay (over time)
            the initial TR length when restarting a trust region.
        switch_strategy_freq: The frequency (in terms of function evaluations) at which
            the strategy should be switched between using a hypervolume objective and
            using random scalarizations.
        tabu_tenure: Number of BO iterations for which a previous X_center is considered
            tabu. A previous X_center is only considered tabu if it was the TR center
            when the TR was terminated.
        fill_strategy: (DEPRECATED) Set to "sobol" to fill trust regions with Sobol
            points until there are at least min_tr_size in each trust region. Set to
            "closest" to include the closest min_tr_size instead. Using "closest" is
            strongly recommended as filling with Sobol points may be very
            sample-inefficient.
        use_noisy_trbo: A boolean denoting whether to expect noisy observations and
            use model predictions for trust region computations.
        use_simple_rff: If True, the GP predictions are replaced with predictions from a
            1-RFF draw during candidate generation.
        batch_limit: default maximum size of joint posterior for TS, lower is less memory intensive,
            while higher is more memory intensive. default: [0,10] (for full posterior sizes) but
            only drawing 10 samples at once.
        use_approximate_hv_computations: Whether to use approximate hypervolume computations. This
            may speed up candidate generation, especially when there are more than 2 objectives.
        approximate_hv_alpha: The value of alpha passed to NondominatedPartitioning. The larger
            the value of alpha is, the faster and less accurate hv computations will be used, see
            NondominatedPartitioning for more details. This parameter only has an effect if
            use_approximate_hv_computations is set to True.
        pred_batch_limit: The maximum batch size to use for `_get_predictions`.
        infer_reference_point: Set this to true if you want to explore the entire Pareto frontier.
            `max_reference_point` will be ignored and we will rely on `infer_reference_point` to infer
            the reference point before generating new candidates.
        fit_gpytorch_options: Options for fitting GPs
        restart_hv_scalarizations: Whether to sample restart points using random scalarizations
    """

    # Base
    dim: int
    batch_size: int
    length: float = 0.8  # length_init
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 10_000  # success_streak
    success_counter: int = 0
    success_tolerance: int = 3  # failure_streak
    best_value: float = -float("inf")
    restart_triggered: bool = False

    # MORBO
    n_trust_regions: int = 5
    eps: float = 1e-3
    use_ard: bool = True
    trim_trace: bool = True

    # Unknown
    qmc: bool = True
    max_tr_size: int = 2000
    min_tr_size: int = 250
    pred_batch_limit: int = 1024
    track_history: bool = True
    fixed_scalarization: bool = False
    n_restart_points: int = 0
    max_reference_point: Optional[List[float]] = None
    hypervolume: bool = True
    winsor_pct: float = 5.0  # this will winsorize the bottom 5%
    decay_restart_length_alpha: float = 0.5
    tabu_tenure: int = 100
    use_noisy_trbo: bool = False
    use_simple_rff: bool = False
    batch_limit: List = None
    use_approximate_hv_computations: bool = False
    approximate_hv_alpha: Optional[float] = None  # Note: Should be >= 0.0
    infer_reference_point: bool = False
    restart_hv_scalarizations: bool = False
    raw_samples: int = 4096  # thompson sampling

    # No desc
    trunc_normal_perturb: bool = False

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


def update_tr_length(state):
    # Update the length of the trust region according to
    # success and failure counters
    # (Just as in original TuRBO paper)
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:  # Restart when trust region becomes too small
        state.restart_triggered = True

    return state


def update_c_state(state, Y_next, C_next):
    # Determine which candidates meet the constraints (are valid)
    bool_tensor = C_next <= 0
    bool_tensor = torch.all(bool_tensor, dim=-1)
    Valid_Y_next = Y_next[bool_tensor]
    Valid_C_next = C_next[bool_tensor]
    if Valid_Y_next.numel() == 0:  # if none of the candidates are valid
        # pick the point with minimum violation
        sum_violation = C_next.sum(dim=-1)
        min_violation = sum_violation.min()
        # if the minimum voilation candidate is smaller than the violation of the incumbent
        if min_violation < state.best_constraint_values.sum():
            # count a success and update the current best point and constraint values
            state.success_counter += 1
            state.failure_counter = 0
            # new best is min violator
            state.best_value = Y_next[sum_violation.argmin()].item()
            state.best_constraint_values = C_next[sum_violation.argmin()]
        else:
            # otherwise, count a failure
            state.success_counter = 0
            state.failure_counter += 1
    else:  # if at least one valid candidate was suggested,
        # throw out all invalid candidates
        # (a valid candidate is always better than an invalid one)

        # Case 1: if the best valid candidate found has a higher objective value that
        # incumbent best count a success, the obj valuse has been improved
        improved_obj = max(Valid_Y_next) > state.best_value + 1e-3 * math.fabs(
            state.best_value
        )
        # Case 2: if incumbent best violates constraints
        # count a success, we now have suggested a point which is valid and thus better
        obtained_validity = torch.all(state.best_constraint_values > 0)
        if improved_obj or obtained_validity:  # If Case 1 or Case 2
            # count a success and update the best value and constraint values
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = max(Valid_Y_next).item()
            state.best_constraint_values = Valid_C_next[Valid_Y_next.argmax()]
        else:
            # otherwise, count a failure
            state.success_counter = 0
            state.failure_counter += 1

    # Finally, update the length of the trust region according to the
    # updated success and failure counters
    state = update_tr_length(state)
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


class SCBO(ContinuousMetaheuristic):
    """Scalable Constrained Bayesian Optimization

    /!\ Works in the unit hypercube. :code:`converter` :ref:`addons` are required.

    See `SCBO <https://botorch.org/tutorials/scalable_constrained_bo>`_.
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
        turbo_state,
        verbose=True,
        surrogate=SingleTaskGP,
        mll=ExactMarginalLogLikelihood,
        likelihood=GaussianLikelihood,
        batch_size=4,
        n_candidates=None,
        initial_size=10,
        beam=2000,
        gpu=False,
        **kwargs,
    ):
        """Short summary.

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space
        turbo_state : CTurboState
            Constrained TuRBO state object dataclass.
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

        assert (
            search_space.loss.constraint is not None
        ), "Loss function must have constraints. `loss.constraint` is None"

        ##############
        # PARAMETERS #
        ##############

        self.surrogate = surrogate
        self.mll = mll
        self.likelihood = likelihood

        self.batch_size = batch_size
        self.n_candidates = n_candidates
        self.initial_size = initial_size

        self.beam = beam

        self.kwargs = kwargs

        #############
        # VARIABLES #
        #############
        self.nconstraint = len(self.search_space.loss.constraint)

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
        self.train_c = torch.empty(
            (0, self.nconstraint),
            dtype=self.dtype,
            device=self.device,
        )

        self.sobol = SobolEngine(dimension=self.search_space.size, scramble=True)

        self._build_kwargs()

        self.cmodels_list = [None] * self.nconstraint

    def _build_kwargs(self):
        self.model_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.surrogate.__init__.__code__.co_varnames
        }

        for m in self.model_kwargs.values():
            if isinstance(m, torch.nn.Module):
                m.to(self.device)

        self.likelihood_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.likelihood.__init__.__code__.co_varnames
        }
        for m in self.likelihood_kwargs.values():
            if isinstance(m, torch.nn.Module):
                m.to(self.device)

        self.mll_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.mll.__init__.__code__.co_varnames
        }
        for m in self.mll_kwargs.values():
            if isinstance(m, torch.nn.Module):
                m.to(self.device)

        print(self.model_kwargs, self.likelihood_kwargs, self.mll_kwargs)

    def _generate_initial_data(self):
        return self.search_space.random_point(self.initial_size)

    #
    def _initialize_model(self, train_x, train_obj, state_dict=None):
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
                num_data=train_x.shape[-2],
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

        return mll, model, train_obj

    def generate_batch(
        self,
        state,
        model,  # GP model
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        batch_size,
        n_candidates,  # Number of candidates for Thompson sampling
        constraint_model,
    ):
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()
        tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)

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
        constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
            model=model, constraint_model=constraint_model, replacement=False
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
        self.train_y = torch.empty((0, 1), dtype=self.dtype, device=self.device)
        self.train_c = torch.empty(
            (0, len(self.search_space.loss.contraint)),
            dtype=self.dtype,
            device=self.device,
        )

    def forward(self, X, Y, constraint):
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
            return train_x, {
                "iteration": self.iteration,
                "algorithm": "InitCTuRBO",
            }
        else:
            torch.cuda.empty_cache()
            if (X is not None and Y is not None) and (len(X) > 0 and len(Y) > 0):
                self.iteration += 1

                new_x = torch.tensor(X, dtype=self.dtype, device=self.device)
                new_obj = torch.tensor(
                    Y, dtype=self.dtype, device=self.device
                ).unsqueeze(-1)
                new_c = torch.tensor(constraint, dtype=self.dtype, device=self.device)

                # update training points
                self.train_x = torch.cat([self.train_x, new_x], dim=0)
                self.train_obj = torch.cat([self.train_obj, new_obj], dim=0)
                self.train_c = torch.cat([self.train_c, new_c], dim=0)

                if len(self.train_x) > self.beam:
                    sidx = torch.argsort(self.train_obj)

                    self.train_x = self.train_x[sidx]
                    self.train_obj = self.train_obj[sidx]
                    self.train_c = self.train_c[sidx]

                    violation = self.train_c.sum(dim=1)
                    nvidx = violation < 0

                    new_x = self.train_x[nvidx][: self.beam]
                    new_obj = self.train_obj[nvidx][: self.beam]
                    new_c = self.train_c[nvidx][: self.beam]

                    if len(new_x) < self.beam:
                        nfill = self.beam - len(new_x)
                        violeted = torch.logical_not(nvidx)
                        v_x = self.train_x[violeted]
                        v_obj = self.train_obj[violeted]
                        v_c = self.train_c[violeted]

                        scidx = torch.argsort(v_c)[nfill]

                        # update training points
                        self.train_x = torch.cat([new_x, v_x[scidx]], dim=0)
                        self.train_obj = torch.cat([new_obj, v_obj[scidx]], dim=0)
                        self.train_c = torch.cat([new_c, v_c[scidx]], dim=0)
                    else:
                        self.train_x = new_x
                        self.train_obj = new_obj
                        self.train_c = new_c

                print(
                    f"TRAIN SIZE: {len(self.train_x)},{len(self.train_obj)},{len(self.train_c)}"
                )
                if len(self.train_obj) < self.initial_size:
                    return [self.search_space.random_point(1)], {
                        "iteration": self.iteration,
                        "algorithm": "AddInitCTuRBO",
                    }
                else:
                    self.turbo_state = update_c_state(
                        state=self.turbo_state, Y_next=new_obj, C_next=new_c
                    )
                    with gpytorch.settings.max_cholesky_size(float("inf")):
                        # reinitialize the models so they are ready for fitting on next iteration
                        # use the current state dict to speed up fitting
                        mll, model, train_Y = self._initialize_model(
                            self.train_x,
                            self.train_obj,
                            state_dict=None,
                        )

                        for i in range(self.nconstraint):
                            cmll, cmodel, _ = self._initialize_model(
                                self.train_x,
                                self.train_c[:, i].unsqueeze(-1),
                                state_dict=None,
                            )
                            try:
                                fit_gpytorch_mll(cmll)
                                self.cmodels_list[i] = cmodel

                            except ModelFittingError:
                                print(
                                    f"In CTuRBO, ModelFittingError for constraint: {self.search_space.loss.constraint[i]}, previous fitted model will be used."
                                )

                        try:
                            fit_gpytorch_mll(mll)
                        except ModelFittingError:
                            return self.search_space.random_point(len(Y)), {
                                "iteration": self.iteration,
                                "algorithm": "FailedCTuRBO",
                            }

                        # optimize and get new observation
                        new_x = self.generate_batch(
                            state=self.turbo_state,
                            model=model,
                            X=self.train_x,
                            Y=train_Y,
                            batch_size=self.batch_size,
                            n_candidates=self.n_candidates,
                            constraint_model=ModelListGP(*self.cmodels_list),
                        )
                        return new_x.cpu().numpy(), {
                            "iteration": self.iteration,
                            "algorithm": "CTuRBO",
                        }
            else:
                return [self.search_space.random_point(1)], {
                    "iteration": self.iteration,
                    "algorithm": "ResampleCTuRBO",
                }

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["cmodels_list"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cmodels_list = [None] * len(self.search_space.loss.constraint)


class MORBO(ContinuousMetaheuristic):
    """Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces

    /!\ Works in the unit hypercube. :code:`converter` :ref:`addons` are required.

    See `MORBO <https://github.com/facebookresearch/morbo>`_.
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

        assert (
            search_space.loss.constraint is not None
        ), "Loss function must have constraints. `loss.constraint` is None"

        ##############
        # PARAMETERS #
        ##############

        self.surrogate = surrogate
        self.mll = mll
        self.likelihood = likelihood

        self.batch_size = batch_size
        self.n_candidates = n_candidates
        self.initial_size = initial_size

        self.kwargs = kwargs

        #############
        # VARIABLES #
        #############
        self.nconstraint = len(self.search_space.loss.constraint)

        self.turbo_state = CTurboState(
            self.search_space.size,
            best_constraint_values=(
                torch.ones(
                    self.nconstraint,
                )
                * torch.inf
            ),
            batch_size=batch_size,
        )

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
        self.train_c = torch.empty(
            (0, self.nconstraint),
            dtype=self.dtype,
            device=self.device,
        )

        self.sobol = SobolEngine(dimension=self.search_space.size, scramble=True)

        self._build_kwargs()

        self.cmodels_list = [None] * self.nconstraint

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

    #
    def _initialize_model(self, train_x, train_obj, state_dict=None):
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

        mll = self.mll(
            model.likelihood,
            model,
            **self.mll_kwargs,
        )

        # load state dict if it is passed
        if state_dict is not None:
            model.load_state_dict(state_dict)

        model.to(self.device)

        return mll, model, train_obj

    def generate_batch(
        self,
        state,
        model,  # GP model
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        batch_size,
        n_candidates,  # Number of candidates for Thompson sampling
        constraint_model,
    ):
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()
        tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)

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
        constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
            model=model, constraint_model=constraint_model, replacement=False
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
        self.train_y = torch.empty((0, 1), dtype=self.dtype, device=self.device)
        self.train_c = torch.empty(
            (0, len(self.search_space.loss.contraint)),
            dtype=self.dtype,
            device=self.device,
        )

    def forward(self, X, Y, constraint):
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
                "algorithm": "InitCTuRBO",
            }
        else:
            if (X is not None and Y is not None) and (len(X) > 0 and len(Y) > 0):
                self.iteration += 1

                X = np.array(X)
                Y = np.array(Y)  # negate Y, BoTorch -> max, Zellij -> min
                constraint = np.array(constraint)

                if len(Y) > 0:
                    new_x = torch.tensor(X, dtype=self.dtype, device=self.device)
                    new_obj = torch.tensor(
                        Y, dtype=self.dtype, device=self.device
                    ).unsqueeze(-1)
                    new_c = torch.tensor(
                        constraint, dtype=self.dtype, device=self.device
                    )

                    self.turbo_state = update_c_state(
                        state=self.turbo_state, Y_next=new_obj, C_next=new_c
                    )

                    # update training points
                    self.train_x = torch.cat([self.train_x, new_x], dim=0)
                    self.train_obj = torch.cat([self.train_obj, new_obj], dim=0)
                    self.train_c = torch.cat([self.train_c, new_c], dim=0)

                    with gpytorch.settings.max_cholesky_size(float("inf")):
                        # reinitialize the models so they are ready for fitting on next iteration
                        # use the current state dict to speed up fitting
                        mll, model, train_Y = self._initialize_model(
                            self.train_x,
                            self.train_obj,
                            state_dict=None,
                        )

                        for i in range(self.nconstraint):
                            cmll, cmodel, _ = self._initialize_model(
                                self.train_x,
                                self.train_c[:, i].unsqueeze(-1),
                                state_dict=None,
                            )
                            try:
                                fit_gpytorch_mll(cmll)
                                self.cmodels_list[i] = cmodel

                            except ModelFittingError:
                                print(
                                    f"In CTuRBO, ModelFittingError for constraint: {self.search_space.loss.constraint[i]}, previous fitted model will be used."
                                )

                        try:
                            fit_gpytorch_mll(mll)
                        except ModelFittingError:
                            return self.sobol.draw(n=len(Y)).cpu().numpy(), {
                                "iteration": self.iteration,
                                "algorithm": "FailedCTuRBO",
                            }

                        # optimize and get new observation
                        new_x = self.generate_batch(
                            state=self.turbo_state,
                            model=model,
                            X=self.train_x,
                            Y=train_Y,
                            batch_size=self.batch_size,
                            n_candidates=self.n_candidates,
                            constraint_model=ModelListGP(*self.cmodels_list),
                        )
                        return new_x.cpu().numpy(), {
                            "iteration": self.iteration,
                            "algorithm": "CTuRBO",
                        }
                else:
                    return self.sobol.draw(n=len(Y)).cpu().numpy(), {
                        "iteration": self.iteration,
                        "algorithm": "ResampleCTuRBO",
                    }
            else:
                return self.sobol.draw(n=1).cpu().numpy(), {
                    "iteration": self.iteration,
                    "algorithm": "ResampleCTuRBO",
                }

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["cmodels_list"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cmodels_list = [None] * len(self.search_space.loss.constraint)
