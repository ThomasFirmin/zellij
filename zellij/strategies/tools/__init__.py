# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-01-13T12:40:11+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

from .chaos_map import Henon, Kent, Logistic, Tent, Random
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

from .geometry import (
    Hypercube,
    Hypersphere,
    Section,
    Direct,
    LatinHypercube,
)

from .tree_search import (
    Breadth_first_search,
    Depth_first_search,
    Best_first_search,
    Beam_search,
    Epsilon_greedy_search,
    Potentially_Optimal_Rectangle,
    Locally_biased_POR,
    Adaptive_POR,
    Soo_tree_search,
    Move_up,
    Cyclic_best_first_search,
)


from .scoring import (
    Min,
    Median,
    Mean,
    Std,
    Distance_to_the_best,
    Distance_to_the_best_centered,
    Belief,
    Nothing,
)

from .measurements import Level, Radius, Sigma2, SigmaInf

from .scoring import (
    Min,
    Median,
    Mean,
    Std,
    Distance_to_the_best,
    Distance_to_the_best_centered,
    Belief,
)
