from abc import abstractmethod
import os
import numpy as np
import pandas as pd


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

    def __init__(self, loss_func, search_space, f_calls, verbose=False):

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
