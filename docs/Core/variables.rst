.. _var:

============
Variables
============

******************
Abstract variables
******************

:ref:`var` functionalities can be extended with :ref:`addons`.

.. automodule:: zellij.core.variables
   :members: Variable
   :undoc-members:
   :show-inheritance:
   :noindex:

**************
Base variables
**************
Basic :ref:`var` are the low-level bricks to define a variable in **Zellij**.

.. automodule:: zellij.core.variables
   :members: IntVar, FloatVar, CatVar
   :undoc-members:
   :show-inheritance:
   :noindex:

~~~~~~~~
Constant
~~~~~~~~
The constant :ref:`var` should not be used.
In particular conditions, it can sometimes replace a regular :ref:`var`,
particularly for decomposition :ref:`meta`.
User should implement directly the constant inside the loss function.

.. automodule:: zellij.core.variables
   :members: Constant
   :undoc-members:
   :show-inheritance:
   :noindex:


******************
Composed variables
******************

Composed :ref:`var` are :ref:`var` made of other :ref:`var`.

.. automodule:: zellij.core.variables
   :members: ArrayVar
   :undoc-members:
   :show-inheritance:
   :noindex:



****************
Future variables
****************

Future :ref:`var`, are variables that are implemented in **Zellij**.
However, the whole package does not yet support them.

.. automodule:: zellij.core.variables
   :members: Block, DynamicBlock
   :undoc-members:
   :show-inheritance:
   :noindex:
