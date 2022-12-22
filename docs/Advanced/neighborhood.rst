.. _nbh:

============
Neighborhood
============

Neighborhood objets are :ref:`addons` defining what a neighborhood is for
a :ref:`var` via :ref:`varadd`, or a :ref:`sp` via :ref:`spadd`.

.. code-block:: python

  from zellij.core import *
  from zellij.core import MockModel
  from zellij.utils.converters import *

  lf = Loss()(MockModel())
  sp = MixedSearchspace(ArrayVar(
                          IntVar("a",-5,5, neighborhood=IntMinmax(2)),
                          FloatVar("b",-5,5, neighborhood=FloatInterval(1.5)),
                          CatVar("b",['f1','f2','f3'], neighborhood=CatInterval()),
                          neighborhood=ArrayInterval()),
                          lf,
                          neighborhood=Intervals())
  point = sp.random_point(1)
  print(point)
  neighbors = sp.to_continuous.convert([[-5,-5],[0,0],point,[5,5]])
  print(float_points)
  int_points = sp.to_continuous.reverse(float_points)
  print(int_points)

#########
Intervals
#########

Intervals :ref:`nbh` are based on drawing a random value between :math:`x \pm neighborhood`.

.. automodule:: zellij.utils.neighborhoods
   :members: Intervals
   :undoc-members:
   :show-inheritance:
   :noindex:


.. automodule:: zellij.utils.neighborhoods
   :members: ArrayInterval, FloatInterval, IntInterval, CatInterval, ConstantInterval
   :undoc-members:
   :show-inheritance:
   :noindex:
