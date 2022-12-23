---
Zellij: Search space update !
---

This release introduces major changes in Zellij. These , concern the package organization, workflow, bug fixes, documentation, and new features. We introduce new ways to define  search spaces and decision variables.

# Breaking Changes

## Package organization

- _zellij/strategies/utils_ > _zellij/strategies/tools_

### Module names

- _fda.py_ -> _dba.py_
- _fractals.py_ -> _geometry.py_
- _heuristics.py_ -> _scoring.py_
- _benchmark.py_ -> _benchmarks.py_

### Class name

- _FDA_ -> _DBA_

### Benchmark functions removed

- shifted_cigar_bent, shifted_rastrigin, alpine, ackley, happycat, shifted_levy, brown, shifted_rotated_rosenbrock, random

### Show method

- All show() methods have been removed from Bayesian optimization, Chaotic optimization, cooling, DBA, Genetic Algorithm, Metaheuristic, PHS, ILS, Simulated Annealing, Searchspace. Plotting methods will be reintroduced in the next Zellij version, in another format.

## Rewrote all heuristics

- See zellij/strategies/tools/scoring.py

## New module _variables.py_
We introduce a new way to better define, in a fine grain, decision  variables:

- ArrayVar: An array of variables
- IntVar: Integer decision variable
- FloatVar: Float decision variable
- CatVar: Categorical variable

Each variable object has its own label, random method and more.

## New module _zellij/core/addons.py_
This module allows to extend Variable and Searchspace functionnalities by dynamically adding addons. Such as converters, neighborhood, operators or distances.

- _zellij/utils/converters.py_: Addons allowing to convert variables. (Categorical to continuous, Continuous to discrete...). See Minmax and Binning converters. It will be used when a Metaheuristic needs to convert a Searchspace to another one.
- _zellij/utils/neighbordhoods.py_: Addons allowing to define what a neighbor is for each variables.
- _zellij/utils/operators.py_: Addons defining Mutation, Crossover, and Selection operators for Genetic Algorithm
- _zellij/utils/distances.py_: Addons defining what a distance for a Searshspace. It will be used when a Metaheuristic needs a distance.
-

## New module _zellij/core/objective.py_
Define the optimization problem. (maximization minimization...) It is a LossFunc parameter. By default Metaheuristics in Zellij are implemented to minimize a loss function. O

- Objective: Abstract class. An Objective targets an outputs from the loss function. it can be an index if the output is a list, or a key if it is a dictionary.
- Maximizer: Define the problem as a maximization one.
- Minimizer: Define the problem as a maximization one
- Lambda: If a given Lambda expression or callable function is given and a list of indexes or keys corresponding to the targeted outputs, it will minimize or maximize it.

## Rewrote Searchspace

A Searchspace is now made of decision variables and a LossFunc.

- Searchspace: it is now an abstract class. We introduce new Searchspace types.
-  MixedSearchspace: Made of an ArrayVar containing other Variables
- ContinuousSearchspace: Made of an ArrayVar containing only FloatVar
- DiscreteSearchspace. Made of an ArrayVar containing only IntVar

# New features

## LossFunc

- kwargs_mode: If True, points will be passed to the original loss function (model) by kwargs. Keys are the decision variables labels.
- only_score: If True, save only the objective value. If False, save the objective value, points, and additional information.
- objective: Determines the problem objective. See _zellij/core/objective.py_.
- MockModel: New class allowing to emulate a  cost-less model for a LossFunc. The user can choose the inputs and outputs formats, and how to compute inputs to obtain desired outputs. It can be used to test scripts, before using the real loss function.

## Searchspace

The whole module has been rewrote. See documentation.

Removed:

- get_neighbor: replaced by _zellij/utils/neighborhoods.py_ addons
- convert_to_continuous: replaced by _zellij/utils/converters.py_ addons
- general_convert: replaced by _zellij/utils/converters.py_ addons
- show
- _create_neighborhood: replaced by _zellij/utils/neighborhoods.py_ addons
- _get_real_neighbor: replaced by _zellij/utils/neighborhoods.py_ addons
- _get_discrete_neighbor: replaced by _zellij/utils/neighborhoods.py_ addons
- _get_categorical_neighbor: replaced by _zellij/utils/neighborhoods.py_ addons

## Geometry

Fractal are now considered as Searchspace.

Entirely rewrote the module:

- parameter _heuristic_: Scoring method. See ./zellij/strategies/tools/scoring.py
- Direct: implements the partition function from Direct
- Soo: implements the partitino function from SOO

## Strategies

All strategies were rewrote to take into account new modules. (Searchspace, decision variables, loss...)

New:

- _zellij/strategies/sampling.py_: Implements new sampling methods for Fractal:Center, Diagonal, Chaos, Chaos_Hypersphere. These inherits from the Metaheuristic abstract class.

## Direct utils
Add _zellij/strategies/tools/direct_utils.py_ module implementing sigma functions used in DIRECT.

## Benchmark

Implements all functions from the SOCO2011 and CEC2020 benchmarks.

## Import shortcuts

- All objects in modules located in _zellij/core/*_ can now be imported using _from zellij.core import ..._
- All strategies can now be imported by _from zellij.strategies import ..._
- All tools can now be imported by _from zellij.strategies.tools import ..._
- All utils can now be imported by _from zellij.utils import ..._

# Documentation update

All the documentation and README has been rewritten. The template has been changed.

# Bug fixes

As many modules were rewrote, many bugs were corrected.

- LossFunc save() method: Fixed an issue where the header was not properly written into the save file.
- LossFunc save() method: Fixed an issue where the objective was saved twice.
- Searchspace: Fixed a bug where _exclude_ was not working.
- MulLogarithmic iteration() method: Fixed a bug where the number of iteration was wrongly computed
- Bayesian Optimization: Fixed an issue were wrong kwargs were passed to the acquisition function
- Genetic Algorithm: Fixed a bug when a filename was given to initalize the population.
- DBA: Fixed a bug where the number of remaining calls to the loss function for the exploration was computed according the exploitation metaheuristic.
- ...
