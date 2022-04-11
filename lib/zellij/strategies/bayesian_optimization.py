from zellij.core.metaheuristic import Metaheuristic
import matplotlib.pyplot as plt

import torch
import numpy as np
from botorch.models import SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import ExpectedImprovement
from botorch import fit_gpytorch_model
import time

import logging

logger = logging.getLogger("zellij.BO")


class Bayesian_optimization(Metaheuristic):
    """Bayesian_optimization

    Bayesian optimization (BO) is a surrogate based optimization method which
    interpolates the actual loss function with a surrogate model, here it is a
    gaussian process. By sampling into this surrogate using a metaheuristic,
    BO determines promising points, which are worth to evaluate with the actual
    loss function. Once done, the gaussian process is updated using results
    obtained by evaluating those encouraging solutions with the loss function.

    Parameters
    ----------
    loss_func : Loss
        Loss function to optimize. must be of type f(x)=y
    search_space : Searchspace
        Search space object containing bounds of the search space
    f_calls : int
        Maximum number of loss_func calls
    verbose : bool
        If False, there will be no print and no progress bar.
    surrogate : botorch.models.model.Model
        Gaussian Process Regressor object from 'botorch'.
        Determines the surrogate model that Bayesian optimization will use to
        interpolate the loss function
    likelihood : gpytorch.mlls
        gpytorch.mlls object it determines which MarginalLogLikelihood to use
        when optimizing kernel's hyperparameters
    acquisition : botorch.acquisition.acquisition.AcquisitionFunction
        An acquisition function or infill criteria, determines how 'promising'
        a point sampled from the surrogate is.
    initial_size : int
        Size of the initial set of solution to draw randomly.
    sampler : botorch.sampling.samplers
        Sampler used for a full Bayesian approach with Monte-Carlo sampling
        applied to approximate the integrated acquisition function.

    gpu: bool
        Use GPU if available

    Attributes
    ----------
    loss_func : Loss
        Loss function to optimize. must be of type f(x)=y
    search_space : Searchspace
        Search space object containing bounds of the search space
    f_calls : int
        Maximum number of loss_func calls
    verbose : bool
        If False, there will be no print and no progress bar.
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
    sampler : botorch.sampling.samplers, default=None
        Sampler used for a full Bayesian approach with Monte-Carlo sampling
        applied to approximate the integrated acquisition function.

    gpu: bool
        Use GPU if available

    """

    def __init__(
        self,
        loss_func,
        search_space,
        f_calls,
        verbose=True,
        surrogate=SingleTaskGP,
        likelihood=ExactMarginalLogLikelihood,
        acquisition=ExpectedImprovement,
        initial_size=10,
        sampler=None,  # Sampler if q-ACQF
        gpu=True,
        **kwargs,
    ):

        super().__init__(loss_func, search_space, f_calls, verbose)

        ##############
        # PARAMETERS #
        ##############

        self.acquisition = acquisition
        self.surrogate = surrogate
        self.likelihood = likelihood
        self.initial_size = initial_size
        self.sampler = sampler

        self.kwargs = kwargs

        #############
        # VARIABLES #
        #############
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = torch.double

        self.bounds = torch.tensor(
            [
                [0.0] * self.search_space.n_variables,
                [1.0] * self.search_space.n_variables,
            ],
            device=self.device,
            dtype=self.dtype,
        )

        self.best_observed = []

        self.iterations = int(
            np.ceil(
                (self.f_calls - self.initial_size) / self.kwargs.get("q", 1)
            )
        )

    def generate_initial_data(self):
        # generate training data
        train_x = torch.rand(
            self.initial_size,
            self.search_space.n_variables,
            device=self.device,
            dtype=self.dtype,
        )
        train_obj = -torch.tensor(
            self.loss_func(
                self.search_space.convert_to_continuous(train_x.numpy(), True)
            )
        ).unsqueeze(
            -1
        )  # add output dimension

        return train_x, train_obj, -self.loss_func.best_score

    def initialize_model(self, train_x, train_obj, state_dict=None):

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

    def optimize_acqf_and_get_observation(self, acq_func, restarts=10, raw=512):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""

        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=self.kwargs.get("q", 1),
            num_restarts=restarts,
            raw_samples=raw,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )

        # observe new values
        new_x = candidates.detach()

        # progress bar
        self.pending_pb(len(new_x))

        new_obj = -torch.tensor(
            self.loss_func(
                self.search_space.convert_to_continuous(new_x.numpy(), True)
            )
        ).unsqueeze(
            -1
        )  # add output dimension

        # progress bar
        self.update_main_pb(
            len(new_x), explor=True, best=self.loss_func.new_best
        )

        return new_x, new_obj

    def run(self, n_process=1, save=False):

        # progress bar
        self.build_bar(self.iterations)

        self.best_observed = []
        self.best_idx = [0]

        # call helper functions to generate initial training data and initialize model
        (
            train_x,
            train_obj,
            best_observed_value,
        ) = self.generate_initial_data()

        # progress bar
        self.pending_pb(self.initial_size)

        mll, model = self.initialize_model(train_x, train_obj)

        # progress bar
        self.update_main_pb(
            self.initial_size, explor=True, best=self.loss_func.new_best
        )
        self.meta_pb.update()

        self.best_observed.append(self.loss_func.best_score)

        iteration = 1

        # run N_BATCH rounds of BayesOpt after the initial random batch
        while (
            iteration < self.iterations and self.loss_func.calls < self.f_calls
        ):

            # fit the models
            fit_gpytorch_model(mll)

            acqf = self.acquisition(
                model=model,
                best_f=-self.loss_func.best_score,
                sampler=self.sampler,
                X_baseline=(train_x,),
                **{
                    key: value
                    for key, value in self.kwargs.items()
                    if key in self.acquisition.__init__.__code__.co_varnames
                },
            )

            # optimize and get new observation
            (
                new_x,
                new_obj,
            ) = self.optimize_acqf_and_get_observation(acqf)

            # update training points
            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])

            # reinitialize the models so they are ready for fitting on next iteration
            # use the current state dict to speed up fitting
            mll, model = self.initialize_model(
                train_x,
                train_obj,
                model.state_dict(),
            )

            # update progress
            self.best_observed.append(self.loss_func.best_score)
            self.best_idx.append(iteration)

            iteration += 1

            # progress bar
            self.meta_pb.update()

        best_idx = np.argpartition(self.loss_func.all_scores, n_process)
        best = [self.loss_func.all_solutions[i] for i in best_idx[:n_process]]
        min = [self.loss_func.all_scores[i] for i in best_idx[:n_process]]

        # self.close_bar()
        return best, min

    def show(self, filename=None):

        if filename == None:
            scores = np.array(self.loss_func.all_scores)
        else:
            data = pd.read_table(filename, sep=",", decimal=".")
            scores = data["loss_value"].to_numpy()

        super().show()

        min = np.argmin(scores)
        indexes = np.repeat(0, self.initial_size)
        indexes = np.append(
            indexes,
            np.repeat(
                np.arange(1, self.iterations, dtype=int),
                self.kwargs.get("q", 1),
            )[: len(scores) - self.initial_size],
        )
        plt.scatter(
            indexes,
            scores,
            c=scores,
            cmap="plasma_r",
        )
        plt.plot(
            self.best_idx,
            self.best_observed,
            color="red",
            lw=2,
            label="Best scores list",
        )
        plt.title("Scores evolution during bayesian optimization")
        plt.scatter(
            min // self.kwargs.get("q", 1),
            scores[min],
            color="red",
            label="Best score",
        )
        plt.annotate(
            str(scores[min]), (min // self.kwargs.get("q", 1), scores[min])
        )
        plt.xlabel("iterations")
        plt.ylabel("Scores")
        plt.legend(loc=1)
        plt.show()

        if filename != None:
            self.search_space.show(
                data.iloc[:, 0 : self.search_space.n_variables], scores
            )
