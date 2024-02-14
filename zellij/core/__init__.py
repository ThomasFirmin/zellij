# Author Thomas Firmin
# Email:  thomas.firmin@univ-lille.fr
# Project: Zellij
# License: CeCILL-C (http://www.cecill.info/index.fr.html)

from .loss_func import Loss, MockModel, MPILoss, SequentialLoss
from .search_space import (
    MixedSearchspace,
    ContinuousSearchspace,
    UnitSearchspace,
    DiscreteSearchspace,
)
from .variables import IntVar, FloatVar, CatVar, ArrayVar, PermutationVar, Variable
from .objective import Minimizer, Maximizer, Lambda, DoNothing

from .experiment import Experiment
from .stop import Calls, Convergence, Combined, Threshold, IThreshold, BooleanStop, Time
from .backup import AutoSave, load_backup

from .metaheuristic import MockMixedMeta
