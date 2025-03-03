.. Satellite Operations Services Optimizer 2025 documentation master file, created by
   sphinx-quickstart on Sun Mar  2 17:47:11 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Satellite Operations Services Optimizer 2025 documentation
==========================================================

Documentation for the Satellite Operations Services Optimizer (SOSO) capstone
project at York University. These pages document the main scheduler of the
project.

There are three main algorithms used in the scheduler.

#. :mod:`soso.genetic` - The genetic algorithm that attempts to optimize the
   schedule based on parameters (like percent of jobs scheduled, equal usage of
   space resources, etc.). Individuals in a population are represented by their
   genome, which in this case is a set of orders that are restricted or
   unrestricted for the individual. Standard genetic algorithm functions are
   used, like fitness evaluation, selection, crossover, and mutation.

#. :mod:`soso.network_flow` - For every generation of the genetic algorithm, the
   network flow algorithm optimizes the individual to find the best possible
   schedule given the individual's set of restricted and unrestricted orders.
   For the network flow algorithm, orders and satellite timeslots are
   represented as nodes in a graph, and a
   `Max-Flow Min-Cut <https://en.wikipedia.org/wiki/Max-flow_min-cut_theorem>`_
   algorithm is used to find the maximum possible number of orders to be
   scheduled.

#. :mod:`soso.bin_packing` - Once the network flow algorithm optimizes a
   schedule, a bin packing algorithm is used to schedule the maximum number of
   downlinks in a schedule. This optimization method models downlinking
   timeslots as bins, whose capacities correspond to their downlinking rates,
   and imaging orders as items to be placed in bins, whose sizes correspond to
   their image sizes. The Google
   `OR-Tools <https://developers.google.com/optimization/pack/bin_packing>`_
   library is used to solve the bin packing problem with linear programming.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   soso
