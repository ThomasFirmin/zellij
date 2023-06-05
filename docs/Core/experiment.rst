.. _exp:


Running experiments
===================

Once you have defined your :ref:`sp`, you can now define the stopping criterion,
and the experiment.

.. automodule:: zellij.core.experiment
   :members: Experiment
   :undoc-members:
   :show-inheritance:
   :noindex:


.. _stop:

Stopping criterions
-------------------

Stopping criterions are :code:`Callable` determining when to stop the :ref:`meta`.

Abstract Class
''''''''''''''

.. automodule:: zellij.core.stop
   :members: Stopping
   :undoc-members:
   :show-inheritance:
   :noindex:

Concrete class
''''''''''''''

.. automodule:: zellij.core.search_space
  :members: Calls, Convergence, Combined
  :undoc-members:
  :show-inheritance:
  :noindex:
