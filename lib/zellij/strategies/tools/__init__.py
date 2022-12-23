# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:37:10+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

from .chaos_map import Henon, Kent, Logistic, Tent, Random
from .tree_search import (
    Breadth_first_search,
    Depth_first_search,
    Best_first_search,
    Beam_search,
    Diverse_best_first_search,
    Cyclic_best_first_search,
    Epsilon_greedy_search,
)
from .cooling import (
    MulExponential,
    MulLogarithmic,
    MulLinear,
    MulQuadratic,
    AddLinear,
    AddQuadratic,
    AddExponential,
    AddTrigonometric,
)
from .scoring import (
    Min,
    Median,
    Mean,
    Std,
    Distance_to_the_best,
    Distance_to_the_best_corrected,
    Belief,
)
