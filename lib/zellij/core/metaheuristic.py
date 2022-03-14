import zellij.utils.progress_bar as pb
from abc import abstractmethod
import os
import numpy as np
import pandas as pd
import enlighten

import logging

logger = logging.getLogger("zellij.Meta")


class Metaheuristic:

    """Metaheuristic

    Metaheuristic is a core object which define the structure of a metaheuristic in zellij.

    Attributes
    ----------

    loss_func : LossFunc
        Loss function to optimize. must be of type f(x)=y

    search_space : Searchspace
        Search space object containing bounds of the search space.

    f_calls : int
        Maximum number of loss_func calls

    save : boolean, optional
        if True save results into a file

    H : Fractal, optional
        If a Fractal is given, allows to use it.

    verbose : boolean, default=True
        Algorithm verbosity

    Methods
    -------
    create_file(self, *args)
        Create a saving file.


    See Also
    --------
    LossFunc : Parent class for a loss function.
    Searchspace : Define what a search space is in Zellij.
    """

    def __init__(self, loss_func, search_space, f_calls, verbose=True):

        ##############
        # PARAMETERS #
        ##############

        self.loss_func = loss_func
        self.search_space = search_space
        self.f_calls = f_calls

        self.verbose = verbose

        #############
        # VARIABLES #
        #############

        # Modify labels in loss func according to SearchSpace labels
        self.loss_func.labels = self.search_space.labels
        # Index of the historic in loss function.
        self.lf_idx = len(self.loss_func.all_scores)

        if self.verbose:
            self.manager = enlighten.get_manager()
        else:
            self.manager = enlighten.get_manager(stream=None, enabled=False)

        self.main_pb = False

    def build_bar(self, total):

        if self.verbose:
            if (not hasattr(self.manager, "zellij_first_line")) or (hasattr(self.manager, "zellij_first_line") and not self.manager.zellij_first_line):

                self.main_pb = True
                self.manager.zellij_first_line = True
                self.best_pb = pb.best_counter(self.manager)
                self.calls_pb_explor, self.calls_pb_exploi, self.calls_pb_pending = pb.calls_counter(self.manager, self.f_calls)

                self.loss_func.manager = self.manager

            else:
                self.main_pb = False
                self.best_pb = False
                self.calls_pb_explor = False
                self.calls_pb_exploi = False
                self.calls_pb_pending = False

        self.meta_pb = pb.metaheuristic_counter(self.manager, total, self.__class__.__name__)

    def update_main_pb(self, nb, explor=True, best=False):
        if self.main_pb and self.verbose:
            if best:
                self.best_pb.update()
            if explor:
                self.calls_pb_explor.update_from(self.calls_pb_pending, nb)
            else:
                self.calls_pb_exploi.update_from(self.calls_pb_pending, nb)

    def pending_pb(self, nb):
        if self.main_pb and self.verbose:
            self.calls_pb_pending.update(nb)

    def close_bar(self):
        if self.main_pb and self.verbose:
            self.best_pb.close()
            self.calls_pb_pending.close()

            self.main_pb = False
            self.manager.zellij_first_line = False

        self.meta_pb.close()

    @abstractmethod
    def run(self):
        pass

    def show(self, filepath="", save=False):

        if filepath:
            all = os.path.join(filepath, "outputs", "all_evaluations.csv")

            all_data = pd.read_table(all, sep=",", decimal=".")
            all_scores = all_data["loss"].to_numpy()
        else:
            all_data = self.loss_func.all_solutions
            all_scores = np.array(self.loss_func.all_scores)

        self.search_space.show(all_data, all_scores, save, self.loss_func.plots_path)

        return all_data, all_scores
