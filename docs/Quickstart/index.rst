==========
Quickstart
==========

In this tutorial, for computation time issues, we will use
the `Himmelblau <https://en.wikipedia.org/wiki/Himmelblau%27s_function>`__
benchmark function.
It is a continuous optimization problem. As a reminder in **Zellij** we consider
by default a minimization problem. See :ref:`objective`.
(e.g. training a neural network is time consumming).
More applications here: :ref:`examples`.

Defining the Loss Function
==========================

A :ref:`lf` is made of a Python
`callable <https://docs.python.org/3/library/functions.html#callable>`_
defined by the user.
In **Zellij**, we consider this function as black-box, it can be whatever
you want, provided it follows a certain pattern.
Indeed, this :ref:`lf` must be of the form :math:`f(x)=y`.
With :math:`x` a set of hyperparameters.
**Zellij** uses a wrapping function called :func:`zellij.core.Loss`
to add features to the user defined function.


Loss function inputs
--------------------
In your defined :ref:`lf`, inputs can be of two types, it can be a list of
hyperparameters.
But, if :code:`kwarg_mode=True` in :func:`zellij.core.Loss`,
then each hyperparameters will be passed as a :code:`kwarg` to the :ref:`lf`.
Each key will be the label of the corresponding :ref:`var`.

Loss function outputs
---------------------
**Zellij** supports alternative output pattern: :math:`f(x)=y,model` for example.
Where:

* :math:`y` can be a `list <https://docs.python.org/3/tutorial/datastructures.html#more-on-lists>`_ or a `dictionary <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`_.
* :math:`model` is optionnal, it is an object with a :code:`save()` method. (e.g. a neural network from Tensorflow or PyTorch)

.. code-block:: python

  import numpy as np
  from zellij.core import Loss

  @Loss(save=False, verbose=True)
  def himmelblau(x):
    x_ar = np.array(x)
    return (x_ar[0] ** 2 + x_ar[1] - 11) ** 2 + (x_ar[0] + x_ar[1] ** 2 - 7) ** 2

  print(himmelblau)


Defining the search space
=========================

Here we will work in 2 dimensions. But **Zellij** supports high dimensional problems.
To define a searchspace one need to define :ref:`var` and a :ref:`lf`.
Available :ref:`var` are:

* **Floats**: :class:`zellij.core.FloatVar` allows to model with upper and lower bounds a float decision variable. You can even change the sampler.
* **Integers**: :class:`zellij.core.IntVar` allows to model with upper and lower bounds a integer decision variable. You can even change the sampler.
* **Categorical**: :class:`zellij.core.CatVar` allows to model a categorical variable with a list of features.
* **Arrays**: :class:`zellij.core.ArrayVar` allows to model an array of :ref:`var`.


.. code-block:: python

  from zellij.core import FloatVar, ArrayVar, ContinuousSearchspace

  values = ArrayVar(FloatVar("float_1", 0,1),FloatVar("float_2", 0,1))
  sp = ContinuousSearchspace(values,himmelblau)

  p1,p2 = sp.random_point(), sp.random_point()
  print(p1)
  print(p2)

Once your search space is defined, you can use some of its functionnalities.
You can draw random points, random attributes...

.. code-block:: python

  rand_att = sp.random_attribute(5)
  rand_pts = sp.random_point(10)

  print(f"Random Attributes: {rand_att}")
  print(f"Random Points: {rand_pts}")

See :ref:`sp` for more information.

Now we can use the loss function and the search space:

.. code-block:: python

  scores = himmelblau(rand_pts)
  print(f"Best solution found:\nf({himmelblau.best_point}) = {himmelblau.best_score}")
  print(f"Number of evaluations:{himmelblau.calls}")
  print(f"All evaluated solutions:{himmelblau.all_solutions}")
  print(f"All loss values:{himmelblau.all_scores}")

  # Reset the loss function for other usage
  himmelblau.reset()

Implementing an optimization strategy
=====================================

Here we will implement a :ref:`bo`, which uses `BoTorch <https://botorch.org/>`_.
In **Zellij** all optimization algorithms are based on the abstract class :ref:`meta`.
An optimization algorithm will be defined by a :ref:`sp`, a :ref:`lf`,
a budget (number of calls to :ref:`lf`).

Here we use an additive exponential cooling schedule.

.. code-block:: python

  from zellij.strategies import Bayesian_optimization

  bo = Bayesian_optimization(sp, 500)

  best, score = bo.run()
  print(f"Best solution found:\nf({best}) = {score}")
