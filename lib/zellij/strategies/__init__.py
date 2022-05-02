from zellij.strategies import utils

__all__ = ["chaos_algorithm", "fda", "genetic_algorithm", "ils", "phs", "sampling", "simulated_annealing"]

__all__.extend(utils.__all__)

# Algorithm
from zellij.strategies.chaos_algorithm import CGS, CLS, CFS, Chaotic_optimization
from zellij.strategies.fda import FDA
from zellij.strategies.genetic_algorithm import Genetic_algorithm
from zellij.strategies.phs import PHS
from zellij.strategies.ils import ILS
from zellij.strategies.simulated_annealing import Simulated_annealing

from zellij.strategies.utils.heuristics import minimum, median, mean, std, dttcb, belief
from zellij.strategies.utils.tree_search import (
    Breadth_first_search,
    Depth_first_search,
    Best_first_search,
    Beam_search,
    Diverse_best_first_search,
    Cyclic_best_first_search,
    Epsilon_greedy_search,
    Potentially_Optimal_Rectangle,
)

from zellij.strategies.utils.chaos_map import Henon, Logistic, Kent, Tent, Random
from zellij.strategies.utils.cooling import (
    MulExponential,
    MulLogarithmic,
    MulLinear,
    MulQuadratic,
    AddLinear,
    AddQuadratic,
    AddExponential,
    AddTrigonometric,
)
