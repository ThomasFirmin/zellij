![alt text](./sources/logo_5st.png)

**Zellij** is an open source Python framework for *HyperParameter Optimization* (HPO) which was orginally dedicated to *Fractal Decomposition based algorithms* [[1]](#1) [[2]](#2).
It includes tools to define mixed search space, manage objective functions, and a few algorithms.
To implements metaheuristics and other optimization methods, **Zellij** uses [DEAP](https://deap.readthedocs.io/)[[3]](#3) for the *Evolutionary Algorithms* part
and [BoTorch](https://botorch.org/) [[4]](#4) for *Bayesian Optimization*.
**Zellij** is defined as an easy to use and modular framework, based on Python object oriented paradigm.

## Install Zellij

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
$ mpiexec -machinefile <path/to/hostfile> -n <number of processes> python3 <path/to/python/script>
```

Be carefull, before using this version, one must be familiar to MPI.

## Quickstart

#### Define your search space
```python

from zellij.core.search_space import Searchspace

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
from zellij.strategies.genetic_algorithm import Genetic_algorithm
ga = Genetic_algorithm(himmelblau, sp, 1000, pop_size=25, generation=40)
ga.run()
ga.show()
```

## Dependencies

#### Original version

* Python >=3.6

* [numpy](https://numpy.org/)
* [DEAP](https://deap.readthedocs.io/en/master/)
* [botorch](https://botorch.org/)
* [gpytorch](https://gpytorch.ai/)
* [matplotlib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)
* [pandas](https://pandas.pydata.org/)

#### Distributed version
* [numpy](https://numpy.org/)
* [DEAP](https://deap.readthedocs.io/en/master/)
* [botorch](https://botorch.org/)
* [gpytorch](https://gpytorch.ai/)
* [matplotlib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)
* [pandas](https://pandas.pydata.org/)
* [mpi4py](https://mpi4py.readthedocs.io/en/stable/)

## Citing

Incomming

## References
<a id="1">[1]</a>
Nakib, A., Ouchraa, S., Shvai, N., Souquet, L. & Talbi, E.-G. Deterministic metaheuristic based on fractal decomposition for large-scale optimization. Applied Soft Computing 61, 468–485 (2017).

<a id="2">[2]</a>
Demirhan, M., Özdamar, L., Helvacıoğlu, L. & Birbil, Ş. I. FRACTOP: A Geometric Partitioning Metaheuristic for Global Optimization. Journal of Global Optimization 14, 415–436 (1999).

<a id="3">[3]</a>
Félix-Antoine Fortin, François-Michel De Rainville, Marc-André Gardner, Marc Parizeau and Christian Gagné, "DEAP: Evolutionary Algorithms Made Easy", Journal of Machine Learning Research, vol. 13, pp. 2171-2175, jul 2012.

<a id="4">[4]</a>
M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. Advances in Neural Information Processing Systems 33, 2020.
