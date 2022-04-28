![alt text](./sources/logo_5st.png)

**Zellij** is an open source Python framework for *HyperParameter Optimization* (HPO) which was orginally dedicated to *Fractal Decomposition based algorithms* [[1]](#1) [[2]](#2).
It includes tools to define mixed search space, manage objective functions, and a few algorithms.
To implements metaheuristics and other optimization methods, **Zellij** uses `DEAP <https://deap.readthedocs.io/>`_ [4]_ for the *Evolutionary Algorithms* part
and `BoTorch <https://botorch.org/>`_ [5]_ for *Bayesian Optimization*.
**Zellij** is defined as an easy to use and modular framework, based on Python object oriented paradigm.

## Installing Zellij

#### Original version
```
$ pip install zellij
```

#### Distributed Zellij

This version requires MPI, such as [MPICH](https://www.mpich.org/) or [Open MPI](https://www.open-mpi.org/).
It based on [mpi4py](https://mpi4py.readthedocs.io/en/stable/intro.html#what-is-mpi)

```
$ pip install zellij[mpi]
```

User will then be able to use the `MPI` option of the `Loss` decorator.
```python
@Loss(MPI=True)
```
Then the python script must be executed using `mpiexec`:
```python
mpiexec -machinefile <path/to/hostfile> -n <number of processes> python3 <path/to/python/script>
```

Be carefull, before using this version, one must be familiar to MPI.

## Quickstart

#### Define your search space
```python

ffrom zellij.core.search_space import Searchspace

labels = ["a","b"]
types = ["R","R"]
values = [[-5, 5],[-5, 5]]
neighborhood = [0.5,0.5]
sp = Searchspace(labels,types,values)
```

#### Define your loss function
```python
import numpy as np
from zellij.core.loss_func import Loss

@Loss(save=False, verbose=True)
def himmelblau(x):
  x_ar = np.array(x)
  return np.sum(x_ar**4 -16*x_ar**2 + 5*x_ar) * (1/len(x_ar))

print(himmelblau)
```

#### Choose an optimization algorithm

```python
ga = Genetic_algorithm(himmelblau, sp, 1000, pop_size=25, generation=40)
ga.run()
ga.show()
```

## Dependencies

#### Original version

* Python >=3.6

* numpy
* DEAP
* botorch
* gpytorch
* matplotlib
* seaborn
* pandas

#### Distributed version
* numpy
* DEAP
* botorch
* gpytorch
* matplotlib
* seaborn
* pandas
* mpi4py

## Citing

```xml
@article{}
```

## References
<a id="1">[1]</a>
Nakib, A., Ouchraa, S., Shvai, N., Souquet, L. & Talbi, E.-G. Deterministic metaheuristic based on fractal decomposition for large-scale optimization. Applied Soft Computing 61, 468–485 (2017).
<a id="1">[2]</a>
Demirhan, M., Özdamar, L., Helvacıoğlu, L. & Birbil, Ş. I. FRACTOP: A Geometric Partitioning Metaheuristic for Global Optimization. Journal of Global Optimization 14, 415–436 (1999).
