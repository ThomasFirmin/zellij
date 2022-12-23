# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-11-09T10:55:33+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

import zellij.utils.progress_bar as pb
from abc import abstractmethod
import os
import numpy as np
import pandas as pd
import enlighten

import logging

logger = logging.getLogger("zellij.meta")


class Metaheuristic(object):

    """Metaheuristic

    :ref:`meta` is a core object which defines the structure
    of a metaheuristic in Zellij. It is an abtract class.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    f_calls : int
        Maximum number of calls to search.space_space.loss.

    save : boolean, optional
        If True save results into a file

    verbose : boolean, default=True
        Activate or deactivate the progress bar.

    See Also
    --------
    :ref:`lf` : Parent class for a loss function.
    :ref:`sp` : Defines what a search space is in Zellij.
    """

    def __init__(self, search_space, f_calls, verbose=True):

        ##############
        # PARAMETERS #
        ##############
        self.search_space = search_space
        self.f_calls = f_calls

        self.verbose = verbose

        #############
        # VARIABLES #
        #############

        if self.verbose:
            self.manager = enlighten.get_manager()
        else:
            self.manager = enlighten.get_manager(stream=None, enabled=False)

        self.main_pb = False

    def build_bar(self, total):
        """build_bar(total)

        build_bar is a method to build a progress bar.
        It is a purely aesthetic feature to get info on the execution.
        You can deactivate it, with `verbose=False`.

        Parameters
        ----------
        total : int
            Length of the progress bar.

        """

        if self.verbose:
            if (not hasattr(self.manager, "zellij_first_line")) or (
                hasattr(self.manager, "zellij_first_line")
                and not self.manager.zellij_first_line
            ):

                self.main_pb = True
                self.manager.zellij_first_line = True
                self.best_pb = pb.best_counter(self.manager)
                (
                    self.calls_pb_explor,
                    self.calls_pb_exploi,
                    self.calls_pb_pending,
                ) = pb.calls_counter(self.manager, self.f_calls)

                self.search_space.loss.manager = self.manager

            else:
                self.main_pb = False
                self.best_pb = False
                self.calls_pb_explor = False
                self.calls_pb_exploi = False
                self.calls_pb_pending = False

        self.meta_pb = pb.metaheuristic_counter(
            self.manager, total, self.__class__.__name__
        )

    def update_main_pb(self, nb, explor=True, best=False):
        """update_main_pb(nb, explor=True, best=False)

        Update the main progress bar with a certain number.

        Parameters
        ----------
        nb : int
            Length of the update. e.g. if the progress bar measure
            the number of iterations, at each iteration `nb=1`.
        explor : bool, default=True
            If True the color associated to the update will be blue.
            Orange, otherwise.
        best : bool default=False
            If True the score of the current solution will be displayed.

        """
        if self.main_pb and self.verbose:
            if best:
                self.best_pb.update()
            if explor:
                self.calls_pb_explor.update_from(self.calls_pb_pending, nb)
            else:
                self.calls_pb_exploi.update_from(self.calls_pb_pending, nb)

    def pending_pb(self, nb):
        """pending_pb(nb)

        Update the progress bar with a pending property (white).
        This update will be replaced when using
        `update_main_pb`.

        Parameters
        ----------
        nb : type
            Length of the pending objects.

        """
        if self.main_pb and self.verbose:
            self.calls_pb_pending.update(nb)

    def close_bar(self):
        """close_bar()

        Delete the progress bar. (must be executed at the end of `run` method)

        """
        if self.main_pb and self.verbose:
            self.best_pb.close()
            self.calls_pb_pending.close()

            self.main_pb = False
            self.manager.zellij_first_line = False

        self.meta_pb.close()

    @abstractmethod
    def run(self):
        """run()

        Abstract method, describes how to run a metaheuristic

        """
        pass
