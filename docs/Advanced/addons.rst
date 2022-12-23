.. _addons:

======
Addons
======
An addons, is an object that is linked to another one. It allows to extend
some functionnalities of the target without modifying its implementation.
The user can graft addons to :ref:`var` and :ref:`sp` by using the kwargs
in the init function.

Known kwargs are:

* For :ref:`sp`:

  * :code:`to_continuous`: used in algorithms using continuous :ref:`sp`, when the given :ref:`sp` is not continuous.

  * :code:`to_discrete`: used in algorithms using discrete :ref:`sp`, when the given :ref:`sp` is not discrete.

  * :code:`distance`: used in algorithms where a distance is needed.

  * :code:`neighbor`: used in algorithms using neighborhood between points.

  * :code:`mutation`: used in :ref:`ga`.

  * :code:`selection`: used in :ref:`ga`.

  * :code:`crossover`: used in :ref:`ga`.


* For :ref:`var`:

  * :code:`to_continuous`: used by the to_continuous :ref:`spadd`.

  * :code:`to_discrete`: used by the to_discrete :ref:`spadd`.

  * :code:`neighbor`: Defines what is a neighbor for a given :ref:`var`. It uses the neighbor :ref:`spadd`.


.. automodule:: zellij.core.addons
   :members: Addon
   :undoc-members:
   :show-inheritance:
   :noindex:

.. _varadd:

##############
Variable Addon
##############

.. automodule:: zellij.core.addons
   :members: VarAddon
   :undoc-members:
   :show-inheritance:
   :noindex:

**********
Subclasses
**********
.. automodule:: zellij.core.addons
  :members: VarNeighborhood, VarConverter
  :undoc-members:
  :show-inheritance:
  :noindex:

.. _spadd:

##################
Search space Addon
##################

.. automodule:: zellij.core.addons
   :members: SearchspaceAddon
   :undoc-members:
   :show-inheritance:
   :noindex:

**********
Subclasses
**********

.. automodule:: zellij.core.addons
   :members: Neighborhood, Converter, Operator, Mutator, Crossover, Selector, Distance
   :undoc-members:
   :show-inheritance:
   :noindex:


########
See also
########

Many addons are linked to metaheuristics.
See:

  * :ref:`gaadd`
  * :ref:`nbh`
  * :ref:`dist`
  * :ref:`convert`
