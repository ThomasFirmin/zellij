# Zellij

## Description

Zellij is an open source Python package for the optimization of expensive black box functions in a high dimensional space. It is based on a fractal decomposition of the search space, and includes diverse exploration and exploitation metaheuristics such as an evolutionary algorithm, a simulated annealing, or chaotic optimization. Moreover Zellij includes tools to model and manipulate \'hypercubic\' non-constrained search space.

## Fractal Decomposition Algorithm

This algorithm is based on a branch and bound strategy to decompose the search space, into subspaces. The algorithm merges various well-known optimization problems:
* **Decomposition problem:**
  * To decompose the search space, selecting the right hypervolume is essential. (hypercube, hypersphere, dividing rectangles...)
* **Exploration problem:**
 * To decompose a fractal, a subspace, the algorithm must determine a heuristic value to each of them. To do so an exploration phase is applied to quickly determine if a fractal is promising or not.
* **Exploitation problem:** 
  * When the algorithm reach the last fractal level, the algorithm applies an intensification phase to exploit the best found solution found into the final fractal.
* **Tree search problem:**
  * By decomposing fractals into subspaces and so on, the algorithm builds a rooted tree. To efficiently explore this graph, tree search algorithm such as Best First Search or Beam Search are used.
* **Rating problem:**
  * Once a fractal has been explored, the algorithm must asign an heuristic value to determine if decomposing this fractal is worth it or not. Zellij can use best value, median, mean, distance to the best found solution...

![Fractal decomposition](https://github.com/ThomasFirmin/zellij/blob/main/sources/fda.PNG?raw=true)
 
## Chaotic Optimization

The algorithm is composed of 3 parts:
* Chaotic Global Search (CGS):
  * CGS is used for the exploitation phase
* Chaotic Local Search (CLS):
  * CLS uses chaos and a progressive zoom to perform an exploitation on the best found solution found by the CGS
* Chaotic Fine Search (CFS):
  * CFS is used as an intensification procedure. It allows to refine the best solution found by the CLS.

During the exploitation phase chaos allows to quickly and violently move over the search space, unlike the exploitation phase where chaos is used to waggle points around an initial solution.

## Install

Download **Zellij 0.0.1** folder and include it to your Python project.

## Code example

```python

from zellij.fda import FDA
from zellij.strategies.chaos_algorithm import CGS
from zellij.strategies.ils import ILS

from zellij.utils.search_space import Searchspace
from zellij.utils.loss_func import loss_func
from zellij.utils.benchmark import himmelblau

# Determine the search space
label = ["a","b"]
type = ["R","R"]
values = [[-5,5],[-5,5]]
neighborhood = [0.5,0.5]
sp = Searchspace(label, type, values, neighborhood)

# Wrap the function to iterate on it, manage its kwargs...
model = loss_func(himmelblau)

# Determine kwargs for the exploration strategy
CGS_kwargs = {"f_calls":100,"level":25,"chaos_map":"henon","create":True}

# Determine kwargs for the exploitation strategy
ILS_kwargs = {"f_calls":1000,"red_rate":0.80,"precision":1e-5}

# Determine tree search kwargs
tree_search = {"beam_length":10}

# Determine hypervolume kwargs
vk = {}

# Construct the fractal decomposition algorithm
sa = FDA(model.evaluate, sp, 20000,CGS,ILS,tree_search="BestFS", heuristic="belief",level=6,volume_kwargs=vk,explor_kwargs=CGS_kwargs,exploi_kwargs=ILS_kwargs,ts_kwargs=tree_search,fractal="hypersphere")

# Run
sa.run()

# Show the results
sa.show()

```

## Parameters

### Fractal Decomposition

Parameters | Type | Description | Default
------------ | ------------- | ------------- | -------------
`loss_func` | `function` | Function that takes a vector of solutions and return an heuristic value | :x:
`search_space` | `Search_space` | Contains lower and upper bounds of the search space and other methods to draw random points or neighbors | :x:
`f_calls` | `int` | Stopping criterion: number of calls to the loss function | :x:
`exploration` | `Metaheuristic` or `list(Metaheuristic)` | Object that contains the exploration strategy and a `run` method, a list of metaheuristics can be passed. If so, at each level the metaheuristic at the current level index is used, if `len(exploration)<level` the last one is used for the next level until the exploration phase.| :x:
`exploitation` | `Metaheuristic` | Object that contains the exploitation strategy and a `run` method | :x:
`fractal` | `string` | Determine the hypervolume to use for the decomposition:`hypercube`,`hypersphere`,`direct` | `"hypersphere"`
`heuristic` | `string` | Determine the method to rate a fractal after an exploration: `best`,`median`,`mean`,`std`,`dttcb`,`belief` | `"best"`
`level` | `int` | Determine the depth of the search space, the fractal depth | `5`
`volume_kwargs` | `dict` | Key word arguments for the selected hypervolume | `{}`
`explor_kwargs` | `dict` or `list(dict)`| Key word arguments for the exploration | `{}`
`exploi_kwargs` | `dict`| Key word arguments for the exploitation | `{}`
`ts_kwargs` | `dict`| Key word arguments for the tree search algorithm | `{}`
`verbose` | `boolean` | If `True` displays information during the execution | `True`

----------------------

### Chaotic Optimization

Parameters | Type | Description | Default
------------ | ------------- | ------------- | -------------
`loss_func` | `function` | Function that takes a vector of solutions and return an heuristic value | :x:
`search_space` | `Search_space` | Contains lower and upper bounds of the search space and other methods to draw random points or neighbors | :x:
`f_calls` | `int` | Stopping criterion: number of calls to the loss function | :x:
`chaos_map` | `string` | Determine the chaos dynamic to use: `henon`,`kent`,`logistic`,`tent` | `"henon"`
`exploration_ratio` | `float` | Determine the ratio between exploration and exploitation | `0.80`
`level` | `tuple` | A tuple of size 3 determining the number of chaotic levels for CGS, CLS and CFS | `(32,8,2)`
`polygon` | `int` | Determine the number of vertex used for the rotating polygon for CLS and CFS | `4`
`red_rate` | 0<`float`<1 | Zoom rate for the CLS and CLS | `0.50`
`verbose` | `boolean` | If `True` displays information during the execution | `True`

----

#### CGS

Parameters | Type | Description | Default
------------ | ------------- | ------------- | -------------
`loss_func` | `function` | Function that takes a vector of solutions and return an heuristic value | :x:
`search_space` | `Search_space` | Contains lower and upper bounds of the search space and other methods to draw random points or neighbors | :x:
`f_calls` | `int` | Stopping criterion: number of calls to the loss function | :x:
`level` | `int` | Number of chaotic levels | :x:
`chaos_map` | `string` | Determine the chaos dynamic to use: `henon`,`kent`,`logistic`,`tent` | :x:

----

#### CLS

Parameters | Type | Description | Default
------------ | ------------- | ------------- | -------------
`loss_func` | `function` | Function that takes a vector of solutions and return an heuristic value | :x:
`search_space` | `Search_space` | Contains lower and upper bounds of the search space and other methods to draw random points or neighbors | :x:
`f_calls` | `int` | Stopping criterion: number of calls to the loss function | :x:
`level` | `tuple` | Number of chaotic levels | :x:
`polygon` | `int` | Determine the number of vertex used for the rotating polygon | :x:
`red_rate` | 0<`float`<1 | Zoom rate | :x:
`chaos_map` | `string` | Determine the chaos dynamic to use: `henon`,`kent`,`logistic`,`tent` | :x:

----

#### CFS

Parameters | Type | Description | Default
------------ | ------------- | ------------- | -------------
`loss_func` | `function` | Function that takes a vector of solutions and return an heuristic value | :x:
`search_space` | `Search_space` | Contains lower and upper bounds of the search space and other methods to draw random points or neighbors | :x:
`f_calls` | `int` | Stopping criterion: number of calls to the loss function | :x:
`chaos_map` | `string` | Determine the chaos dynamic to use: `henon`,`kent`,`logistic`,`tent` | :x:
`exploration_ratio` | `float` | Determine the ratio between exploration and exploitation | :x:
`level` | `tuple` | Number of chaotic levels | :x:
`polygon` | `int` | Determine the number of vertex used for the rotating polygon | :x:
`red_rate` | 0<`float`<1 | Zoom rate | :x:

----------------------

### Genetic Algorithm

Parameters | Type | Description | Default
------------ | ------------- | ------------- | -------------
`loss_func` | `function` | Function that takes a vector of solutions and return an heuristic value | :x:
`search_space` | `Search_space` | Contains lower and upper bounds of the search space and other methods to draw random points or neighbors | :x:
`f_calls` | `int` | Stopping criterion: number of calls to the loss function | :x:
`pop_size` | `int` | Population size | `10`
`generation` | `int` | Number of generation | `1000`
`verbose` | `boolean` | If `True` displays information during the execution | `True`

----------------------

### Simulated Annealing

Parameters | Type | Description | Default
------------ | ------------- | ------------- | -------------
`loss_func` | `function` | Function that takes a vector of solutions and return an heuristic value | :x:
`search_space` | `Search_space` | Contains lower and upper bounds of the search space and other methods to draw random points or neighbors | :x:
`f_calls` | `int` | Stopping criterion: number of calls to the loss function | :x:
`max_iter` | `int` | Number of iteration after each temperature decrease | :x:
`T_0` | `float` | Initial temperature | :x:
`T_end` | `float` | Final temperature | :x:
`n_peaks` | `int` | Number of violent temperature increase when `T_end` is reached | `1`
`red_rate` | `float` | reduction rate of the temperature | `0.80`
`verbose` | `boolean` | If `True` displays information during the execution | `True`

----------------------

### Hypersphere Heuristic Search (FDA)

Parameters | Type | Description | Default
------------ | ------------- | ------------- | -------------
`loss_func` | `function` | Function that takes a vector of solutions and return an heuristic value | :x:
`search_space` | `Search_space` | Contains lower and upper bounds of the search space and other methods to draw random points or neighbors | :x:
`f_calls` | `int` | Stopping criterion: number of calls to the loss function | :x:

----------------------

### Intensive Local Search (FDA)

Parameters | Type | Description | Default
------------ | ------------- | ------------- | -------------
`loss_func` | `function` | Function that takes a vector of solutions and return an heuristic value | :x:
`search_space` | `Search_space` | Contains lower and upper bounds of the search space and other methods to draw random points or neighbors | :x:
`f_calls` | `int` | Stopping criterion: number of calls to the loss function | :x:
`red_rate` | `float` | Step reduction between two iterations | `0.50`
`precision` | `float` | Stopping criterion, when `red_rate` < `precision` | `1e-5`

### Build your own exploration, exploitation and tree search algorithms

## Citing

```xml
@article{}
```

## Sources

* Nassime Aslimani, El-Ghazali Talbi, Rachid Ellaia. Tornado: An Autonomous Chaotic Algorithm for Large Scale Global Optimization. 2020
* Léo Souquet, Amir Nakib, El-Ghazali Talbi. Deterministic multi-objective fractal decomposition algorithm. MIC 2019 - 13th Metaheuristics International Conference, Jul 2019, Cartagena, Colombia. ⟨hal-02304975⟩

