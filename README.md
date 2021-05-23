# Chaotic Optimisation

## Description

This Python package includes 2 efficients meta-heuristics for large scale global optimization and high dimensional continuous search spaces. Both algorithms include parametrable exploration and exploitation strategies.

* **Chaotic Optimization Algorithm**
  * Uses a chaotic dynamic to efficiently find local optima
* **Fractal Decomposition Algorithm**
  * Decompose the search space using various fractals (hypersphere, hypercube...) to select and exploit promising areas.

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
------------ | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
`loss_func` | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | `function` | Function that takes a vector of float and return a loss value (float) | :x:
`lo_bounds` | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | `list(float)` | Lower bounds of each dimension of the search space | :x:
`up_bounds` | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | `list(float)` | Upper bounds of each dimension of the search space | :x:
`gds` | :heavy_check_mark: | :x: | :x: | :x: | `Boolean` | Use an adaptative gradient descent search after CLS and CFS | `False`
`N_symetric_p` | :heavy_check_mark: | :x: | :heavy_check_mark: | :heavy_check_mark: | `int` | Determine the number of points for the rotating polygon. (4 square, 5 pentagon...) | `8`
`choas_map_func` | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | `string | list(string)` | Define the chaotic map to use. If a list of maps is given, it shuffle the different maps. | `"henon_map"`
`f_call` | :heavy_check_mark: | :x: | :x: | :x: | `int` | Determine the number of loss function calls. | `1000`
`M_global` | :heavy_check_mark: | :x: | :x: | :x: | `int` | Determine the global number of iteration | `200`
`M_local` | :heavy_check_mark: | :x: | :x: | :x: | `int` | Determine the number of exploitation iteration | `50`
`N_level_cgs` | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | `int` | Determine the number of chaos level for CGS | `50`
`N_level_cls` | :heavy_check_mark: | :x: | :heavy_check_mark: | :x: | `int` | Determine the number of chaos level for CLS | `5`
`N_level_cfs` | :heavy_check_mark: | :x: | :x: | :heavy_check_mark: | `int` | Determine the number of chaos level for CFS | `5`
`red_rate` | :heavy_check_mark: | :x: | :heavy_check_mark: | :heavy_check_mark: | `float` | Determine the zoom rate for CLS and CFS | `0.5`
`windowed_cgs`| :heavy_check_mark: | :x: | :x: | :x: | `float` | If > 0, the CGS uses a decreasing centered zoom over the search space | `0`
`penalize` (not yet implemented) | :heavy_check_mark: | :x: | :x: | :x: | `Boolean` | Penalize loss function and area of the search space to avoid overlapping points | `0`
`return_history`| :heavy_check_mark: | :x: | :x: | :x: | `Boolean` | If True return the history: `[function calls, penalized points, points, values, colors, size, best point]`  | `True`

