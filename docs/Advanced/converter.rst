.. _convert:

==========
Converters
==========

Converters are :ref:`addons` for:ref:`sp` and :ref:`var`, they allow to convert
points into another format (e.g. mixed to continuous).
Each :ref:`var` must have their own converter implemented, as well as
the :ref:`sp` so it can convert points according each :ref:`var`.

.. code-block:: python

  from zellij.core import IntVar, ArrayVar, DiscreteSearchspace, Loss
  from zellij.core import MockModel
  from zellij.utils.converters import IntMinmax,ArrayMinmax, Continuous

  lf = Loss()(MockModel())
  sp = DiscreteSearchspace(ArrayVar(
                          IntVar("a",-5,5, to_continuous=IntMinmax()),
                          IntVar("b",-5,5, to_continuous=IntMinmax()),
                          to_continuous=ArrayMinmax()),
                          lf,
                          to_continuous=Continuous())
  point = sp.random_point(1)
  print(point)
  float_points = sp.to_continuous.convert([[-5,-5],[0,0],point,[5,5]])
  print(float_points)
  int_points = sp.to_continuous.reverse(float_points)
  print(int_points)

#############
To continuous
#############

.. automodule:: zellij.utils.converters
  :members: Continuous
  :undoc-members:
  :show-inheritance:
  :noindex:

******
MinMax
******

.. automodule:: zellij.utils.converters
   :members: ArrayMinmax, FloatMinmax, IntMinmax, CatMinmax, ConstantMinmax
   :undoc-members:
   :show-inheritance:
   :noindex:

###########
To discrete
###########

.. automodule:: zellij.utils.converters
  :members: ArrayBinning, FloatBinning, IntBinning, CatBinning, ConstantBinning
  :undoc-members:
  :show-inheritance:
  :noindex:

*******
Binning
*******

.. automodule:: zellij.utils.converters
  :members: Discrete
  :undoc-members:
  :show-inheritance:
  :noindex:


#######
Others
#######

.. automodule:: zellij.utils.converters
  :members: DoNothing
  :undoc-members:
  :show-inheritance:
  :noindex:
