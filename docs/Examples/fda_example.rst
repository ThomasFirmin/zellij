===============
FDA with Zellij
===============

This is a corrected implementation of FDA [1]_. The distance to the best, that one
can find in the original paper is replaced by a corrected version.

In *Zellij*, FDA is decomposed as follow:

* **Geometry**: Hypersphere
* **Tree search**: MoveUp (sorted Depth First Search)
* **Exploration**: Promising Hypersphere Search (PHS)
* **Exploitation**: Intensive Local Search (ILS)
* **Scoring**: Corrected Distance to the best

.. [1] A. Nakib, S. Ouchraa, N. Shvai, L. Souquet, and E.-G. Talbi, ‘Deterministic metaheuristic based on fractal decomposition for large-scale optimization’, Applied Soft Computing, vol. 61, pp. 468–485, Dec. 2017, doi: 10.1016/j.asoc.2017.07.042.

.. code-block:: python

  from zellij.utils.benchmarks import himmelblau
  from zellij.core import ArrayVar, FloatVar, Loss, Experiment, Threshold
  from zellij.strategies import PHS, ILS, DBA
  from zellij.strategies.tools import Hypersphere, Distance_to_the_best, Move_up
  from zellij.utils.converters    import FloatMinmax, ArrayConverter, Basic

  lf = Loss(save=True)(himmelblau)
  values = ArrayVar(
      FloatVar("float_1", -5 , 5, converter=FloatMinmax()),
      FloatVar("float_2", -5, 5, converter=FloatMinmax()),
      converter=ArrayConverter(),
  )
  sp = Hypersphere(values, lf, converter=Basic())

  explor = PHS(sp, inflation=1.75)
  exploi = ILS(sp, inflation=1.75)
  stop1 = Threshold(None, "current_calls", 3)  # set target to None, DBA will automatically asign it.
  stop2 = Threshold(None,"current_calls", 100)  # set target to None, DBA will automatically asign it.
  dba = DBA(sp, Move_up(sp,5),(explor,stop1), (exploi,stop2),scoring=Distance_to_the_best())

  stop3 = Threshold(lf, "calls",1000)
  exp = Experiment(dba, stop3, save="exp_fda")
  exp.run()
  print(f"Best solution:f({lf.best_point})={lf.best_score}")



  import pandas as pd
  import matplotlib.pyplot as plt
  import numpy as np

  data = pd.read_csv("exp_direct/outputs/all_evaluations.csv")
  print(data)

  fig, ax = plt.subplots()
  x = y = np.linspace(-5, 5, 100)
  X,Y = np.meshgrid(x,y)
  Z = (X ** 2 + Y - 11) ** 2 + (X + Y ** 2 - 7) ** 2


  map = ax.contourf(X,Y,Z,cmap="plasma", levels=100)
  fig.colorbar(map)

  plt.scatter(data["float_1"],data["float_2"],c="cyan",s=0.1)
  plt.plot()

.. image:: ../sources/fda_himmel.png
  :width: 2400
