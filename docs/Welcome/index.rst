Zellij Documentation
====================

**Zellij** is an open source Python framework for *HyperParameter Optimization* (HPO) which was originally
dedicated to *Decomposition based algorithms* [1]_ [2]_.
It includes tools to define mixed search space, manage objective functions, and a few algorithms.
To implement metaheuristics and other optimization methods, **Zellij** uses `DEAP <https://deap.readthedocs.io/>`__ [3]_
for the *Evolutionary Algorithms* part and `BoTorch <https://botorch.org/>`__ [4]_ for *Bayesian Optimization*.

In **Zellij** we consider a **minimization** problem with a loss function:

:math:`f:\mathcal{X} \subset \mathbb{R}^n \rightarrow \mathbb{R}: \hat{x} = \mathrm{argmin}_{x \in \mathcal{X}}f(x)`.

With, :math:`\hat{x}` the global optima, :math:`f` the objective function, and :math:`\mathcal{X}` a compact set made of inequalities (e.g. upper and lower bounds of decision variables).

Currently, **Zellij** supports mono-objective problems and *mixed, non-constrained and non-dynamic* search spaces.

Hyperparameter optimization problem
-----------------------------------

A hyperparameter in machine learning is a parameter set before the learning phase, those features will impact the learning and the model design. Whereas, parameters, such as synaptic weights in a neural network, are learned.
Hyperparameters can be of multiple types, real, discrete, ordinal, categorical, binary... For example, the learning rate of a Stochastic Gradient Descent, the filter size for a convolution,
the activation function for a group of neurons...

When doing HPO of neural networks, one hhasave to face several challenges:

* **Black box** loss functions: the only knowledge is the inputs and outputs of the function. One cannot compute the derivatives or the Jacobian matrix for example.
* **Expensive** loss functions: the evaluation takes minutes up to days.
* **Noisy** loss function: the value of :math:`f(x)` with :math:`x` fixed is not necessary the same over multiple runs.
* **High dimensional and mixed** search space.

Popular algorithms for such problems are *Grid Search*,
*Random Search*, *Genetic Algorithms** or *Bayesian Optimization*.

Defining the search space is a critical phase. Selecting too many
hyperparameters can result to a combinatorial explosion.
Moreover, selecting non-significant hyperparameters will have a low impact on the
training or model.
Therefore, before running any optimization algorithms, one should be careful about
the search space design. Optimizing in a poorly defined search space will lead
to worthless optimization.

Install Zellij
--------------

Basic version
^^^^^^^^^^^^^

To install the base version of **Zellij**, you can use:

.. code-block:: bash

  pip install zellij

Distributed Zellij
^^^^^^^^^^^^^^^^^^

This version requires a MPI library, such as `MPICH <https://www.mpich.org/>`__
or `Open MPI <https://www.open-mpi.org/>`__.
It is based on `mpi4py <https://mpi4py.readthedocs.io/en/stable/intro.html#what-is-mpi>`__.

.. code-block:: bash

  pip install zellij[mpi]

User will then be able to use the :code:`MPI=True` option of :func:`zellij.core.Loss`.

Then the python script must be executed using :code:`mpiexec`:

.. code-block:: bash

  mpiexec -machinefile <path/to/hostfile> -n <number of processes> python3 <path/to/python/script>

First steps
-----------

.. code-block:: Python

  from zellij.core import FloatVar, ArrayVar, ContinuousSearchspace, Loss, Experiment, Threshold
  from zellij.strategies import Bayesian_optimization
  from zellij.utils.benchmarks import himmelblau

  # Decision variable
  values = ArrayVar(FloatVar("float_1", -5,5),FloatVar("float_2", -5,5))

  # Loss function
  lf = Loss(save=False)(himmelblau)

  # Search space
  sp = ContinuousSearchspace(values,lf)

  # Optimization Algorithm
  bo = Bayesian_optimization(sp, initial_size = 30)

  # Stopping criterion
  stop = Threshold(lf, "calls", 100)

  # Create an experiment
  exp = Experiment(bo, stop)

  # Run the experiment
  exp.run()
  print(f"Best solution found:\nf({lf.best_point}) = {lf.best_score}")

Dependencies
------------

* **Python** >=3.6
* `numpy <https://numpy.org/>`__>=1.21.4
* `DEAP <https://deap.readthedocs.io/en/master/>`__>=1.3.1
* `botorch <https://botorch.org/>`__>=0.6.3.1
* `gpytorch <https://gpytorch.ai/>`__>=1.6.0
* `pandas <https://pandas.pydata.org/>`__>=1.3.4
* `scipy <https://scipy.org/>`__>=1.9.3
* [mpi]: `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`__>=3.1.2

Contributors
------------
* Thomas Firmin: thomas.firmin@univ-lille.fr
* El-Ghazali Talbi: el-ghazali.talbi@univ-lille.fr

References
----------
.. [1] Nakib, A., Ouchraa, S., Shvai, N., Souquet, L. & Talbi, E.-G. Deterministic metaheuristic based on fractal decomposition for large-scale optimization. Applied Soft Computing 61, 468–485 (2017).
.. [2] Demirhan, M., Özdamar, L., Helvacıoğlu, L. & Birbil, Ş. I. FRACTOP: A Geometric Partitioning Metaheuristic for Global Optimization. Journal of Global Optimization 14, 415–436 (1999).
.. [3] Félix-Antoine Fortin, François-Michel De Rainville, Marc-André Gardner, Marc Parizeau and Christian Gagné, "DEAP: Evolutionary Algorithms Made Easy", Journal of Machine Learning Research, vol. 13, pp. 2171-2175, jul 2012.
.. [4] M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. Advances in Neural Information Processing Systems 33, 2020
