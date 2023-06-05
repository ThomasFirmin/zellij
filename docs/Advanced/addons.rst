.. _addons:

======
Addons
======
An :ref:`addons`, is an object linked to another one. It allows extending
some functionalities of the target without modifying its implementation.
The user can graft :ref:`addons` to :ref:`var` and :ref:`sp` by using the :code:`kwargs`
in the :code:`init` function.

Known kwargs are:

* For :ref:`sp`:

  * :code:`converter`: used to convert a solution from a space to another.

  * :code:`distance`: used in algorithms where a distance is needed.

  * :code:`neighbor`: used in algorithms using neighborhood between points.

  * :code:`mutation`: used in :ref:`ga`.

  * :code:`selection`: used in :ref:`ga`.

  * :code:`crossover`: used in :ref:`ga`.


* For :ref:`var`:

  * :code:`converter`: used by the converter from :ref:`spadd`.

  * :code:`neighbor`: Defines what a neighbor is for a given :ref:`var`. It uses the neighbor :ref:`spadd`.


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
