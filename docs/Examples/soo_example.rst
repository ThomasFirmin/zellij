===============
SOO with Zellij
===============

This is an implementation of SOO [1]_ with *Zellij*, applied on the 2D Himmelblau function.

In *Zellij*, SOO is decomposed as follow:

* **Geometry**: Trisection (SOO)
* **Tree search**: Best subset at each level
* **Exploration**: Center of hyper-rectangles
* **Exploitation**: No exploitation strategy applied
* **Scoring**: Minimum

.. [1] R. Munos, ‘Optimistic Optimization of a Deterministic Function without the Knowledge of its Smoothness’, p. 9.

.. warning:: 
  The following code is deprecated.
  
.. code-block:: python

  from zellij.core.geometry import Soo
  from zellij.strategies import DBA
  from zellij.strategies.tools.tree_search import Soo_tree_search
  from zellij.strategies.tools.scoring import Distance_to_the_best_corrected

  from zellij.core import ContinuousSearchspace, FloatVar, ArrayVar, Loss
  from zellij.utils.benchmarks import himmelblau

  loss = Loss()(himmelblau)
  values = ArrayVar(
                    FloatVar("a",-5,5),
                    FloatVar("b",-5,5)
                    )

  def SOO_al(
    values,
    loss,
    calls,
    verbose=True,
    level=600,
    section=3,
    force_convert=False,
    ):

    sp = Soo(
        values,
        loss,
        calls,
        force_convert=force_convert,
        section=section,
    )

    dba = DBA(
        sp,
        calls,
        tree_search=Soo_tree_search(sp, level),
        verbose=verbose,
    )
    dba.run()

    return sp

  sp = SOO_al(values, loss, 1000)
  best = (sp.loss.best_point, sp.loss.best_score)
  print(f"Best solution found:f({best[0]})={best[1]}")

  import matplotlib.pyplot as plt
  import numpy as np

  fig, ax = plt.subplots()
  x = y = np.linspace(-5, 5, 100)
  X,Y = np.meshgrid(x,y)
  Z = (X**4-16*X**2+5*X + Y**4-16*Y**2+5*Y)/2

  map = ax.contourf(X,Y,Z,cmap="plasma", levels=100)
  fig.colorbar(map)
  ax.scatter(
              np.array(sp.loss.all_solutions)[:,0],
              np.array(sp.loss.all_solutions)[:,1],
              s=1,
              label="Points"
            )
  ax.scatter(
              best[0][0],
              best[0][1],
              c="red",
              s=5,
              label="Best"
            )
  ax.set_title("SOO on 2D Himmelblau function")
  ax.legend()
  plt.show()

.. image:: ../sources/soo_himmel.png
  :width: 2400
