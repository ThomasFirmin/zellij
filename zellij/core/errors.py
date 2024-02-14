# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

import logging

logger = logging.getLogger("zellij.error")


class DimensionalityError(Exception):

    """
    Raised when dimensionality of two objects mismatch.
    """

    pass


class TargetError(Exception):
    """
    Raised when the target of an instance mismatch.
    """

    pass


class ModelError(Exception):
    """
    Raised when a model returned by :ref:`lf` mismatch requirements.
    """


class OutputError(Exception):
    """
    Raised when an error occurs on the outputs of loss function.
    """


class InputError(Exception):
    """
    Raised when an error occurs on the inputs of a function.
    """


class InitializationError(Exception):
    """
    Raised when a target of an addon is still set to None, when getter is called.
    """


class UnassignedProcess(Exception):
    """
    Raised when a process is not assigned to any role.
    """
