==================
DIRECT with Zellij
==================

This implementation of DIRECT [1]_ is based on the DIRECT Optimization Algorithm User Guide [2]_.
DIRECT is a special case of *fractal decomposition based* algorithm, as the way it decomposes the search space does .. note::
fit with the *Zellij* definition of a fractal. Indeed, here the partition,
the exploration and the scoring are the same function. DIRECT requires to
sample each center of future subset, before concretely creating them.

In *Zellij*, DIRECT is decomposed as follow:

* **Geometry**: DIRECT (Partition, sample and score at the same time)
* **Tree search**: Potentially Optimal Rectangle
* **Exploration**: Done by the geometry
* **Exploitation**: No exploitation strategy used
* **Scoring**: Minimum (Done by the geometry)
* *Sigma* : :math:`\sigma^2` (proper to DIRECT)

.. [1] D. R. Jones, C. D. Perttunen, and B. E. Stuckman, ‘Lipschitzian optimization without the Lipschitz constant’, J Optim Theory Appl, vol. 79, no. 1, pp. 157–181, Oct. 1993, doi: 10.1007/BF00941892.
.. [2] Finkel, Daniel E.. “Direct optimization algorithm user guide.” (2003).

.. warning:: 
  The following code is deprecated.

.. code-block:: python

  <code>

  from zellij.core import FloatVar, ArrayVar, Loss
  from zellij.strategies import DBA
  from zellij.strategies.tools import Direct, Potentially_Optimal_Rectangle, Sigma2, SigmaInf, Min

  from zellij.utils.benchmarks import himmelblau

  loss = Loss()(himmelblau)
  values = ArrayVar(
                    FloatVar("a",-5,5),
                    FloatVar("b",-5,5)
                    )
  # Define DIRECT algorithm
  def Direct_al(
    values,
    loss,
    calls,
    verbose=True,
    level=600,
    error=1e-4,
    maxdiv=3000,
    force_convert=False,
  ):

    sp = Direct(
        values,
        loss,
        scoring=Min(),
        sigma=Sigma2(len(values)),
    )

    dba = DBA(
        sp,
        tree_search=Potentially_Optimal_Rectangle(
            sp, level, error=error, maxdiv=maxdiv
        ),
        verbose=verbose,
    )

    stop = Threshold(loss, 'calls', calls)
    exp = Experiment(dba, stop)
    exp.run()
    
    return sp

  # Run DIRECT, and get initial search space
  sp = Direct_al(values, loss, 1000)
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
  ax.set_title("DIRECT on 2D Himmelblau function")
  ax.legend()
  plt.show()

.. image:: ../sources/direct_himmel.png
  :width: 2400
