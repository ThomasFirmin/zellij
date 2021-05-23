# Chaotic Optimisation

## Description

This Python package includes 2 efficients meta-heuristics for large scale global optimization and high dimensional continuous search spaces. Both algorithms include parametrable exploration and exploitation strategies.

* **Chaotic Optimization Algorithm**
  * Uses a chaotic dynamic to efficiently find local optima
* **Fractal Decomposition Algorithm**
  * Decomposes the search space using various fractals (hypersphere, hypercube...) to select and exploit promising areas.

## Install

### Requirements

**chaotic-optimization** requires Numpy for fast computation and Matplotlib for plotting results.

### Lated version
Download chaotic-optimization folder and include it into your Python project. You can then import FDA or chaotic-optimization according to your needs.


## Chaotic Optimization Algorithm

### Description

Chaotic Optimization Algorithm includes 3 strategies which use a chaotic dynamic, symmetrization and leveling at various scale.
* **Chaotic Global Search (CGS)**: Explore the search space using a chaotic map. Chaos is used to violently agitate the points distribution over search space.
* **Chaotic Local Search (CLS)**: It is an exploitation-based algorithm. It explores the neighborhood from an initial solution. It uses a chaos map to gently wiggle points. By moving iteratively from best solution to another and by using a zoom, **CLS** allows to search over different promising search space.
* **Chaotic Fine Search (CFS)**: It is an exploitation-based algorithm to intensify the exploitation of the best solution found by the **CLS**. **CFS** uses an adaptative zoom to intensify the search in the neighborhood of a the best solution.

All three strategies are used in the Tornado algorithm. But in they can also be used independently.

### Parameters


Parameters | Tornado | CGS | CLS | CFS | Type | Description | Default
------------ | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
`loss_func` | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | `function` | Function that takes a vector of float (points) and return a loss value (float) | :x:
`lo_bounds` | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | `list(float)` | Lower bounds of each dimension of the search space | :x:
`up_bounds` | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | `list(float)` | Upper bounds of each dimension of the search space | :x:
`N_symetric_p` | :heavy_check_mark: | :x: | :heavy_check_mark: | :heavy_check_mark: | `int` | Determine the number of points for the rotating polygon. (4 square, 5 pentagon...) | `8`
`chaos_map_func` | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | `string or list(string)` | Define the chaotic map to use. If a list of maps is given, it shuffles the different maps. `henon_map, kent_map, logistic_map, tent_map` | `"henon_map"`
`f_call` | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | `int` | Determine the number of loss function calls. | `1000`
`M_global` | :heavy_check_mark: | :x: | :x: | :x: | `int` | Determine the global number of iteration | `200`
`M_local` | :heavy_check_mark: | :x: | :x: | :x: | `int` | Determine the number of exploitation iteration | `50`
`N_level_cgs` | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | `int` | Determine the number of chaos level for CGS | `50`
`N_level_cls` | :heavy_check_mark: | :x: | :heavy_check_mark: | :x: | `int` | Determine the number of chaos level for CLS | `5`
`N_level_cfs` | :heavy_check_mark: | :x: | :x: | :heavy_check_mark: | `int` | Determine the number of chaos level for CFS | `5`
`red_rate` | :heavy_check_mark: | :x: | :heavy_check_mark: | :heavy_check_mark: | `float` | Determine the zoom rate for CLS and CFS | `0.5`
`gds` | :heavy_check_mark: | :x: | :x: | :x: | `Boolean` | Use an adaptative gradient descent search after CLS and CFS | `False`
`windowed_cgs`| :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | `float` | If > 0, the CGS uses a decreasing centered zoom over the search space | `0`
`penalize` (not yet implemented) | :heavy_check_mark: | :x: | :x: | :x: | `Boolean` | Penalize loss function and area of the search space to avoid overlapping points | `0`
`return_history`| :heavy_check_mark: | :x: | :x: | :x: | `Boolean` | If True return the history: `[function calls, penalized points, points, values, colors, size, best point]`  | `True`

### Code example

```python

import numpy as np
import tornado

def himmelblau(y):
    return np.sum(y**4 -16*y**2 + 5*y) * (1/len(y))
  
dim=200
M_global = 500
M_local = 50
N_cgs = 10
N_cls = 5
N_cfs = 5
N_p = 8

lo_bounds = np.array([-5 for i in range(dim)])
up_bounds = np.array([5 for i in range(dim)])

tornado = tornado.Tornado(himmelblau, lo_bounds, up_bounds, chaos_map_func = "kent_map", M_global = M_global, M_local = M_local, N_level_cgs = N_cgs, N_level_cls = N_cls, N_level_cfs = N_cfs, N_symetric_p = N_p, return_history = False)
 
best_point = tornado.run()

```
### Figures

![Chaotic optimization with himmelblau](https://github.com/ThomasFirmin/chaotic-optimisation/figures/main/tornado.jpg?raw=true)

## Fractal Decomposition Algorithm

### Description

Fractal Decomposition Algorithm is method which decomposes the search space using various fractals (hypersphere, hypercube...) to select and exploit promising areas. It uses an exploitation, exploration and a search tree algorithms. Each fractal is scored according to the quality of the solution found by the exploitation phase, the best one, the father, is selected and decomposed, and so on. At the final level, best fractals are exploited. Father and children fractals builda tree, to search into this tree the algorithm usesa tree search algorithm (Best First Search, Beam Search,...)

### 3 algorithms

* Exploration: exploration is used to explore the fractal and score it.
* Exploitation: exploitation is used at the final level on the best fractals
* Tree search algorithm: used to efficiently search over the tree composed of fractals

### Parameters

Parameters | Type | Description | Default
------------ | ------------- | ------------- | -------------
`loss_func` |  `function` | Function that takes a vector of float (points) and return a loss value (float) | :x:
`lo_bounds` | `list(float)` | Lower bounds of each dimension of the search space | :x:
`up_bounds` | `list(float)` | Upper bounds of each dimension of the search space | :x:
`f_call` | `int` | Determine the number of loss function calls. | `1000`
`fractal` | `string` | Determine the hypervolume to use as fractal. `hypercube,hypersphere` | `hypersphere`
`exploration` | `function` | Determine the exploration algorithm to use. See >insert< for more info | `LHS`
`exploitation` | `function` | Determine the exploitation algorithm to use. See >insert< for more info | `ILS`
`level` | `int` | Determine the level of the fractal decomposition (tree depth) | `5`
`tree_search` | `string` | Determine the search tree algorithm. `BFS,DFS,BS,BestFS` See >insert< for more info| `DFS`
`inflation` | `float` | Determine the inflation rate of the hypervolume.| `1.75`

### Code example

```python

import numpy as np
from fda import *
from fda_func import *

def himmelblau(y):
    return np.sum(y**4 -16*y**2 + 5*y) * (1/len(y))

f = FDA(lo_bounds,up_bounds,himmelblau, loss_call=10000, level=4,inflation=1,exploration=CGS,exploitation=CLS,tree_search="BestFS")
hypersphere = f.run()
    
```

### Figures

![Chaotic optimization with himmelblau](https://github.com/ThomasFirmin/chaotic-optimisation/figures/main/fda.jpg?raw=true)

### Build your own exploration, exploitation and tree search algorithm

#### Exploration and exploitation procedures

Exploration and exploitation algorithms must take as parameters an hypervolume `H` and a loss function `loss_func`. It must also compute the loss function and update the hypervolume with the computed points and score using `add_point(score,point, color="black")` method. `color` is only used for ploting. The function must return the number of loss calls.

##### Code example

```python

import numpy as np
from fractal import Fractal

# Lazy Hypervolume Search
def LHS(H,loss_func):
    loss_call = 0

    for i in range(H.dim):
        inf = np.copy(H.center)
        sup = np.copy(H.center)

        inf[i] = np.max([H.center[i]-H.radius[i]/np.sqrt(H.dim),H.lo_bounds[i]])
        sup[i] = np.min([H.center[i]+H.radius[i]/np.sqrt(H.dim),H.up_bounds[i]])

        score1 = loss_func(inf)
        score2 = loss_func(sup)

        loss_call += 2
        H.add_point(score1, inf,"blue")
        H.add_point(score2, sup,"blue")

    return loss_call
    
```

#### Tree search algorithm (NOT FINISHED, EXPERIMENTAL)

A tree search algorithm uses:
* Open list: contains all unexplored nodes
* Closed list: contains all explored nodes

#### Code example

```python

import numpy as np
from tree_search import Tree_search

class Breadth_first_search(Tree_search):

    def __init__(self,open,n_fractal):

        super().__init__(open)

        self.n_fractal = n_fractal

        self.actual_level = self.open[0].level

        self.level_size = self.n_fractal**(self.actual_level+1)
        self.next_frontier = []

        self.beam_length = 1

    def add(self,c):

        self.next_frontier.append(c)

        if len(self.next_frontier) == self.level_size:

            self.actual_level = c.level

            self.level_size = self.n_fractal**(self.actual_level+1)

            self.open = sorted(self.next_frontier,reverse=self.reverse,key= lambda x: x.score)[:]

            self.next_frontier = []

    def get_next(self):

        if len(self.open) > 0:

            if len(self.open) < self.beam_length:
                idx = len(self.open)
            else:
                idx = self.beam_length

            self.how_many = idx
            self.close += self.open[:idx]

            for _ in range(idx):
                self.open.pop(0)

            return True,self.close[-idx:]

        else:
            return False,-1

```

## Citing

## Sources
