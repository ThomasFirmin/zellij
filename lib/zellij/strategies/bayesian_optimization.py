# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-11-09T12:40:37+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.metaheuristic import Metaheuristic

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


class Bayesian_optimization(Metaheuristic):
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
    f_calls : int
        Maximum number of :ref:`lf` calls
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
    gpu: bool, default=True
        Use GPU if available
    **kwargs
        Key word arguments linked to the surrogate and the acquisition function.

    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is
    :ref:`lf` : Describes what a loss function is in Zellij
    :ref:`sp` : Describes what a loss function is in Zellij

    Examples
    --------
    >>> from zellij.core import Loss
    >>> from zellij.core import ContinuousSearchspace
    >>> from zellij.core import FloatVar, ArrayVar
    >>> from zellij.utils.benchmark import himmelblau
    >>> from zellij.strategies.bayesian_optimization import Bayesian_optimization
    >>> import botorch
    >>> import gpytorch
    ...
    >>> lf = Loss()(himmelblau)
    >>> sp = ContinuousSearchspace(ArrayVar(FloatVar("a",-5,5), FloatVar("b",-5,5)),lf)
    >>> bo = Bayesian_optimization(sp, 500,
    ...       acquisition=botorch.acquisition.monte_carlo.qExpectedImprovement,
    ...       q=5)
    >>> bo.run()
    """

    def __init__(
        self,
        search_space,
        f_calls,
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
        f_calls : int
            Maximum number of :ref:`lf` calls
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
        gpu: bool, default=True
            Use GPU if available
        **kwargs
            Key word arguments linked to the surrogate and the acquisition function.
        """

        super().__init__(search_space, f_calls, verbose)

        ##############
        # PARAMETERS #
        ##############

        assert hasattr(search_space, "to_continuous") or isinstance(
            search_space, ContinuousSearchspace
        ), logger.error(
            f"""If the `search_space` is not a `ContinuousSearchspace`,
            the user must give a `Converter` to the :ref:`sp` object
            with the kwarg `to_continuous`"""
        )

        self.acquisition = acquisition
        self.surrogate = surrogate
        self.likelihood = likelihood
        self.initial_size = initial_size
        self.sampler = sampler

        self.kwargs = kwargs

        #############
        # VARIABLES #
        #############
        if gpu:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = "cpu"

        self.dtype = torch.double

        if isinstance(self.search_space, ContinuousSearchspace):

            self.bounds = torch.tensor(
                [
                    [v.low_bound for v in self.search_space.values],
                    [v.up_bound for v in self.search_space.values],
                ],
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self.bounds = torch.tensor(
                [
                    [0.0] * self.search_space.size,
                    [1.0] * self.search_space.size,
                ],
                device=self.device,
                dtype=self.dtype,
            )

        self.iterations = int(
            np.ceil(
                (self.f_calls - self.initial_size) / self.kwargs.get("q", 1)
            )
        )

    def _generate_initial_data(self):
        # generate training data
        train_x = torch.rand(
            self.initial_size,
            self.search_space.size,
            device=self.device,
            dtype=self.dtype,
        )
        if isinstance(self.search_space, ContinuousSearchspace):
            res = self.search_space.loss(
                train_x.cpu().numpy(),
                algorithm="BO",
                acquisition=0,
            )
        else:
            res = self.search_space.loss(
                self.search_space.to_continuous.reverse(train_x.cpu().numpy()),
                algorithm="BO",
                acquisition=0,
            )
        train_obj = torch.tensor(res).unsqueeze(-1)  # add output dimension

        return train_x, train_obj

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

    def _optimize_acqf_and_get_observation(
        self, acq_func, restarts=10, raw=512
    ):
        """Optimizes the acquisition function, and returns a new candidate
        and a noisy observation."""

        # optimize
        candidates, acqf = optimize_acqf(
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
        if isinstance(self.search_space, ContinuousSearchspace):
            res = self.search_space.loss(
                new_x.cpu().numpy(),
                algorithm="BO",
                acquisition=acqf.cpu().item(),
            )
        else:
            res = self.search_space.loss(
                self.search_space.to_continuous.reverse(new_x.cpu().numpy()),
                algorithm="BO",
                acquisition=acqf.cpu().item(),
            )
        new_obj = torch.tensor(res).unsqueeze(-1)  # add output dimension

        # progress bar
        self.update_main_pb(
            len(new_x), explor=True, best=self.search_space.loss.new_best
        )

        return new_x, new_obj

    def _build_acqf_kwarg(self):
        self.acqf_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.acquisition.__init__.__code__.co_varnames
        }

    def run(self, H=None, n_process=1):

        """run(n_process=1)

        Runs BO.

        Parameters
        ----------
        H : Fractal, optional
            When used by :ref:`dba`, a fractal corresponding to the current subspace is given

        n_process : int, default=1
            Determine the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the :code:`n_process` best found points
            to the continuous format.

        best_scores : list[float]
            Returns a list of the :code:`n_process` best found scores associated
            to best_sol.

        """
        if H:
            sp = H
        else:
            sp = self.search_space

        # progress bar
        self.build_bar(self.iterations)

        # call helper functions to generate initial training data and initialize model
        train_x, train_obj = self._generate_initial_data()

        # progress bar
        self.pending_pb(self.initial_size)

        mll, model = self._initialize_model(train_x, train_obj)

        # progress bar
        self.update_main_pb(
            self.initial_size, explor=True, best=self.search_space.loss.new_best
        )
        self.meta_pb.update()

        iteration = 1

        # run N_BATCH rounds of BayesOpt after the initial random batch
        while (
            iteration < self.iterations
            and self.search_space.loss.calls < self.f_calls
        ):

            # fit the models
            fit_gpytorch_model(mll)

            # Add potentially usefull kwargs for acqf kwargs
            self.kwargs["best_f"] = -self.search_space.loss.best_score
            self.kwargs["X_baseline"] = (train_x,)

            # Build acqf kwargs
            self._build_acqf_kwarg()
            acqf = self.acquisition(model=model, **self.acqf_kwargs)

            # optimize and get new observation
            (
                new_x,
                new_obj,
            ) = self._optimize_acqf_and_get_observation(acqf)

            # update training points
            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])

            # reinitialize the models so they are ready for fitting on next iteration
            # use the current state dict to speed up fitting
            mll, model = self._initialize_model(
                train_x,
                train_obj,
                model.state_dict(),
            )

            iteration += 1

            # progress bar
            self.meta_pb.update()

        self.close_bar()
        return self.search_space.loss.get_best(n_process)
