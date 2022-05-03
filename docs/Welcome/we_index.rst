====================
Zellij Documentation
====================

**Zellij** is an open source Python framework for *HyperParameter Optimization* (HPO) which was orginally dedicated to *Fractal Decomposition based algorithms* [1]_ [2]_ [3]_ .
It includes tools to define mixed search space, manage objective functions, and a few algorithms.
To implements metaheuristics and other optimization methods, **Zellij** uses `DEAP <https://deap.readthedocs.io/>`_ [4]_ for the *Evolutionary Algorithms* part
and `BoTorch <https://botorch.org/>`_ [5]_ for *Bayesian Optimization*.

In **Zellij** we consider a  **minimization** problem with a loss function :math:`f: \mathcal{X} \subset \mathbb{R}^n \rightarrow \mathbb{R}`:

.. math::

      \hat{x} = \mathrm{argmin}_{x \in \mathcal{X}}f(x)

With, :math:`\hat{x}` the global optima, :math:`f` the objective function, and :math:`\mathcal{X}` a compact set made of inequalities (e.g. upper and lower bounds of decision variables).

Currently **Zellij** supports mono-objective problems and *mixed, non-constrained and non-dynamic* search spaces.

Hyperparameter optimization problem
=======================================

A hyperparameter in machine learning is a parameter set before the learning phase, those features will impact the learning and the model design. Whereas, parameters, such as synaptic weights in a neural network, are learned.
Hyperparameters can be of multiple types, real, discrete, ordinal, categorical, binary... For example, the learning rate of a Stochastic Gradient Descent, the filter size for a convolution,
the activation function for a group of neurons...

When doing HPO of neural networks, one have to face several challenges:

* **Black box** loss functions: the only knowledge are the inputs and ouputs of the function. One cannot computes the derivatives or the Jacobian matrix for example.
* **Expensive** loss functions: the evaluation takes minutes up to days.
* **Noisy** loss function: the value of :math:`f(x)` with :math:`x` fixed is not necessary the same over multiple runs.
* **High dimensional and mixed** search space.

Popular algorithms for such problems are *Grid Search*, *Random Search*, *Genetic Algorithms** or *Bayesian Optimization*.

Defining the search space is a critical phase. Selecting too much hyperparameters can result to a combinatorial explosion.
Moreover selecting non significant hyperparameters will have a low impact on the training or model.
Therefore, before running any optimization algorithms, one should be carefull on the search space design. Optimizing in a poorly defined search space will lead to worthless optimization.


Balancing exploration and exploitation
========================================

* **Exploration** (Diversification) is the ability of an algorithm to gather significant information about the search space
* **Exploitation** (Intensifivation) is the ability of an algorithm to use gathered information to exploit promising areas of the search space.

The balance between exploration and exploitation is a main challenge when designing an optimization algorithm. A lack of solution refinement can occur when there is too much exploration.
Whereas an algorithm can be stuck into local optima if there is no exploration.

Some algorithms focused on exploitation are called **Local search** algorithms, such as *Simulated annealing* or *Tabu Search*.


Future features
========================================
* Multi-objective optimization
* Constrained optimization
* Parallel computing with MPI and multi-threading/processing. (Distributed metaheuristics, BO...)
* Enhanced plotting (t-SNE, PCA, parallel plot...)
* New algorithms (CMA-ES, Tabu search, multi-fidelity BO)
* Enhanced searchspaces and conversion to continuous methods.


References
==================
.. [1] Nakib, A., Ouchraa, S., Shvai, N., Souquet, L. & Talbi, E.-G. Deterministic metaheuristic based on fractal decomposition for large-scale optimization. Applied Soft Computing 61, 468–485 (2017).
.. [2] Demirhan, M., Özdamar, L., Helvacıoğlu, L. & Birbil, Ş. I. FRACTOP: A Geometric Partitioning Metaheuristic for Global Optimization. Journal of Global Optimization 14, 415–436 (1999).
.. [3] T.Firmin, Fractal decomposition: A divide and conquer approach for global optimization
.. [4] Félix-Antoine Fortin, François-Michel De Rainville, Marc-André Gardner, Marc Parizeau and Christian Gagné, "DEAP: Evolutionary Algorithms Made Easy", Journal of Machine Learning Research, vol. 13, pp. 2171-2175, jul 2012.
.. [5] M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. Advances in Neural Information Processing Systems 33, 2020
