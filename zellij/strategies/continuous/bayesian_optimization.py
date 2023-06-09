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
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import ExpectedImprovement
from botorch import fit_gpytorch_model
from zellij.core.search_space import ContinuousSearchspace

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
        likelihood=ExactMarginalLogLikelihood,
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
        **kwargs
            Key word arguments linked to the surrogate and the acquisition function.
        """

        super().__init__(search_space, verbose)

        ##############
        # PARAMETERS #
        ##############

        self.acquisition = acquisition
        self.surrogate = surrogate
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

    def _generate_initial_data(self):
        # generate training data
        train_x = torch.rand(
            self.initial_size,
            self.search_space.size,
            device=self.device,
            dtype=self.dtype,
        )

        return train_x

    def _initialize_model(self, train_x, train_obj, state_dict=None):
        # define models for objective and constraint
        model = self.surrogate(
            train_x,
            train_obj,
            **{
                key: value
                for key, value in self.kwargs.items()
                if key in self.surrogate.__init__.__code__.co_varnames
            },
        ).to(train_x)

        mll = self.likelihood(model.likelihood, model)

        # load state dict if it is passed
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return mll, model

    def _optimize_acqf_and_get_observation(self, acq_func, restarts=10, raw=512):
        """Optimizes the acquisition function, and returns a new candidate
        and a noisy observation."""

        bounds = torch.tensor(
            [
                torch.tensor(
                    self.search_space.lower, device=self.device, dtype=self.dtype
                ),
                torch.tensor(
                    self.search_space.upper, device=self.device, dtype=self.dtype
                ),
            ],
            device=self.device,
            dtype=self.dtype,
        )

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

    def _build_acqf_kwarg(self):
        self.acqf_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.acquisition.__init__.__code__.co_varnames
        }

    def reset(self):
        """reset()

        reset :code:`Bayesian_optimization` to its initial state.

        """
        self.initialized = False
        self.train_x = torch.empty((0, self.search_space.size))
        self.train_y = torch.empty((0, 1))
        self.state_dict = None

    def forward(self, X, Y):
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

            return train_x, {"acquisition": 0, "algorithm": "BO"}
        else:
            self.iteration += 1

            new_x = torch.tensor(X)
            new_obj = torch.tensor(Y).unsqueeze(-1)

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
        # run N_BATCH rounds of BayesOpt after the initial random batch
        # fit the models
        fit_gpytorch_model(mll)

        # Add potentially usefull kwargs for acqf kwargs
        self.kwargs["best_f"] = -self.search_space.loss.best_score
        self.kwargs["X_baseline"] = (self.train_x,)

        # Build acqf kwargs
        self._build_acqf_kwarg()
        acqf = self.acquisition(model=model, **self.acqf_kwargs)

        # optimize and get new observation
        new_x, acqf_value = self._optimize_acqf_and_get_observation(acqf)

        return new_x.cpu().numpy(), {
            "acquisition": acqf_value.cpu().item(),
            "algorithm": "BO",
        }
