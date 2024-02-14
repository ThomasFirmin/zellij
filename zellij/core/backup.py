# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

import threading
import time
import os
import pickle


class AutoSave(object):
    def __init__(self, experiment):
        self.experiment = experiment

        self._timer = None
        self.interval = self.experiment.backup_interval
        self.is_running = False
        self.next_call = time.time()

        if self.experiment.backup_interval:
            self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.experiment.backup()

    def start(self):
        if not self.is_running:
            self.next_call += self.interval
            self._timer = threading.Timer(self.next_call - time.time(), self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        if self.experiment.backup_interval:
            self._timer.cancel()  # type: ignore
            self.is_running = False


def load_backup(path, loss):
    """
    Load a saved backup.

    Parameters
    ----------
    path : str
        Folder path of the saved experiment.
    loss : Callable
        Initial loss function (not wrapped by :ref:`lf`). See :ref:`lf`.
    """
    backup = os.path.join(os.path.join(path, "backup"), "experiment.p")
    if os.path.isfile(backup):
        exp = pickle.load(open(backup, "rb"))
        exp.meta.search_space.loss.model = loss
        return exp
    else:
        FileNotFoundError(f"No backup is available at {path}")
