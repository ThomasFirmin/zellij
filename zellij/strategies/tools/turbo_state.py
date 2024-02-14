# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations

from typing import Optional, Union
from abc import ABC, abstractmethod

from zellij.core.errors import InitializationError

import torch
from botorch.acquisition.objective import (
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.generation.sampling import MaxPosteriorSampling
from torch import Tensor


import torch
from dataclasses import dataclass
from torch import Tensor
import numpy as np

import logging
import math

logger = logging.getLogger("zellij.turbotools")


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 1.0
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
    length: float = 1.0
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

    def reset(self):
        self.best_constraint_values *= torch.inf
        self.length = 0.8
        self.length_min = 0.5**7
        self.length_max = 1.6
        self.failure_counter = 0
        self.success_counter = 0
        self.success_tolerance = 3  # Note: The original paper uses 3
        self.best_value = -float("inf")
        self.restart_triggered = False

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


class ConstrainedCostAwareMaxPosteriorSampling(MaxPosteriorSampling):
    r"""Constrained Cost Aware max posterior sampling.

    Posterior sampling where we try to maximize an objective function while
    simulatenously satisfying a set of constraints c1(x) <= 0, c2(x) <= 0,
    ..., cm(x) <= 0 where c1, c2, ..., cm are black-box constraint functions.
    Each constraint function is modeled by a seperate GP model. We follow the
    procedure as described in https://doi.org/10.48550/arxiv.2002.08526.

    Example:
        >>> CMPS = ConstrainedMaxPosteriorSampling(
                model,
                constraint_model=ModelListGP(cmodel1, cmodel2),
            )
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = CMPS(X, num_samples=5)
    """

    def __init__(
        self,
        model: Model,
        cost_model: Model,
        constraint_model: Union[ModelListGP, MultiTaskGP],
        temperature: float,
        best_cost: float,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        replacement: bool = True,
        epsilon: float = 1e3,
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: An optional PosteriorTransform for the objective
                function (corresponding to `model`).
            replacement: If True, sample with replacement.
            constraint_model: either a ModelListGP where each submodel is a GP model for
                one constraint function, or a MultiTaskGP model where each task is one
                constraint function. All constraints are of the form c(x) <= 0. In the
                case when the constraint model predicts that all candidates
                violate constraints, we pick the candidates with minimum violation.
        """
        if objective is not None:
            raise NotImplementedError(
                "`objective` is not supported for `ConstrainedMaxPosteriorSampling`."
            )

        super().__init__(
            model=model,
            objective=objective,
            posterior_transform=posterior_transform,
            replacement=replacement,
        )
        self.cost_model = cost_model
        self.constraint_model = constraint_model
        self.temperature = temperature
        print(f"TEMPERATURE : {temperature}")
        self.best_cost = best_cost
        self.epsilon = epsilon

    def _convert_samples_to_scores(
        self, Y_samples, Cost_samples, C_samples, num_samples
    ) -> Tensor:
        r"""Convert the objective and constraint samples into a score.

        The logic is as follows:
            - If a realization has at least one feasible candidate we use the objective
                value as the score and set all infeasible candidates to -inf.
            - If a realization doesn't have a feasible candidate we set the score to
                the negative total violation of the constraints to incentivize choosing
                the candidate with the smallest constraint violation.

        Args:
            Y_samples: A `num_samples x batch_shape x num_cand x 1`-dim Tensor of
                samples from the objective function.
            Cost_samples: A `num_samples x batch_shape x num_cand x 1`-dim Tensor of
                samples from the cost model.
            C_samples: A `num_samples x batch_shape x num_cand x num_constraints`-dim
                Tensor of samples from the constraints.

        Returns:
            A `num_samples x batch_shape x num_cand x 1`-dim Tensor of scores.
        """
        is_feasible = (C_samples <= 0).all(
            dim=-1
        )  # num_samples x batch_shape x num_cand
        has_feasible_candidate = is_feasible.any(dim=-1)

        scores = Y_samples.clone()
        scores[~is_feasible] = -float("inf")
        if not has_feasible_candidate.all():
            # Use negative total violation for samples where no candidate is feasible
            total_violation = (
                C_samples[~has_feasible_candidate]
                .clamp(min=0)
                .sum(dim=-1, keepdim=True)
            )
            scores[~has_feasible_candidate] = -total_violation

        # Filter according to cost
        delta_cost = Cost_samples - self.best_cost
        is_costly = delta_cost > 0

        # normalize costly
        if torch.any(is_costly):
            max_costly = delta_cost[is_costly].max()
            norm_delta_costly = delta_cost[is_costly] / max_costly
            non_costly = scores.shape[1] - len(norm_delta_costly)

            # Compute acceptance probability
            probabilities = torch.exp(-norm_delta_costly / self.temperature)
            random_sample = torch.rand(
                len(norm_delta_costly), device=probabilities.device
            )
            deny = random_sample > probabilities
            deny_sum = deny.sum()
            non_deny = len(norm_delta_costly) - deny_sum

            # Keep costly samples according to probability
            # if number kept elements are lower than batch size
            # slowly force the acceptance
            eps = 0
            while non_costly + non_deny < num_samples:
                print("DENYING")
                eps += self.epsilon
                random_sample = (
                    torch.rand(len(norm_delta_costly), device=probabilities.device)
                    - eps
                )
                deny = random_sample > probabilities
                deny_sum = deny.sum()
                non_deny = len(norm_delta_costly) - deny_sum

            scores[:, is_costly][:, deny] = -float("inf")
            print(f"DENIED : {deny_sum} {deny.shape}")
        else:
            print(Cost_samples)
            print("NOTHING COSTLY")
        return scores

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
                `X[..., i, :]` is the `i`-th sample.
        """
        posterior = self.model.posterior(
            X=X,
            observation_noise=observation_noise,
            # Note: `posterior_transform` is only used for the objective
            posterior_transform=self.posterior_transform,
        )
        Y_samples = posterior.rsample(sample_shape=torch.Size([num_samples]))

        # Constraints
        c_posterior = self.constraint_model.posterior(
            X=X, observation_noise=observation_noise
        )
        C_samples = c_posterior.rsample(sample_shape=torch.Size([num_samples]))

        # Cost
        Cost_samples = self.cost_model(X=X)

        # Convert the objective and constraint samples into a scalar-valued "score"
        scores = self._convert_samples_to_scores(
            Y_samples=Y_samples,
            C_samples=C_samples,
            Cost_samples=Cost_samples,
            num_samples=num_samples,
        )
        return self.maximize_samples(X=X, samples=scores, num_samples=num_samples)


class Temperature(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def temperature(self, elapsed: float) -> float:
        pass


class SkewedBell(Temperature):
    """SkewedBell

    SkewedBell cooling for CASCBO.

    :math:`T = \\gamma\\frac{xe^{\\sqrt{\\alpha x^p}}}{x+e^{-\\alpha x^p}}`

    Attributes
    ----------
    gamma : float
        Influences the value of the peak.
    alpha : float
        Skewness of the curve
    p : float
        Influences the shift of the curve.

    Methods
    -------
    cool()
        Decrease temperature and return the current temperature.
    reset()
        Reset cooling schedule
    iterations()
        Get the theoretical number of iterations to end the schedule.

    """

    def __init__(self, gamma: float, alpha: float, p: float):
        self.gamma = gamma
        self.alpha = alpha
        self.p = p

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value: float):
        if value > 0:
            self._gamma = value
        else:
            raise InitializationError(f"gamma must be >0. Got {value}")

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        if value > 0:
            self._alpha = value
        else:
            raise InitializationError(f"alpha must be >0. Got {value}")

    @property
    def p(self) -> float:
        return self._p

    @p.setter
    def p(self, value: float):
        if value > 0:
            self._p = value
        else:
            raise InitializationError(f"p must be >0. Got {value}")

    def temperature(self, elapsed: float) -> float:
        n = elapsed * np.exp(-np.sqrt(self.alpha) * elapsed**self.p)
        d = elapsed + np.exp(-self.alpha * elapsed**self.p)
        return self.gamma * n / d
