.. _tree_search:

======================
Tree search algorithms
======================

The :ref:`tree_search` module defines how to expand the tree of :ref:`frac`.
It is used by :ref:`dba`.

.. automodule:: zellij.strategies.tools.tree_search
   :members: Tree_search
   :undoc-members:
   :show-inheritance:
   :noindex:


******
Basics
******

.. automodule:: zellij.strategies.tools.tree_search
   :members: Breadth_first_search, Depth_first_search, Best_first_search
   :undoc-members:
   :show-inheritance:
   :noindex:

********
Advanced
********

.. automodule:: zellij.strategies.tools.tree_search
   :members: Epsilon_greedy_search, Diverse_best_first_search
   :undoc-members:
   :show-inheritance:
   :noindex:

********
Prunning
********

.. automodule:: zellij.strategies.tools.tree_search
  :members: Beam_search, Cyclic_best_first_search
  :undoc-members:
  :show-inheritance:
  :noindex:

*************
Miscellaneous
*************

.. automodule:: zellij.strategies.tools.tree_search
  :members: Potentially_Optimal_Rectangle, Locally_biased_POR, Adaptive_POR, Soo_tree_search, Move_up
  :undoc-members:
  :show-inheritance:
  :noindex:
