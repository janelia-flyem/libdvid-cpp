.. libdvid documentation master file, created by
   sphinx-quickstart on Sun Apr  3 18:17:21 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The ``libdvid`` Python module implements bindings to the
`libdvid-cpp <https://github.com/janelia-flyem/libdvid-cpp>`_
library to provide convenient access to selected portions of the
`DVID <https://github.com/janelia-flyem/dvid>`_ REST API to Python code. 

Installation
============

.. code-block:: bash

   conda install -c flyem -c conda-forge libdvid-cpp

Tutorials
=========

- `Basic Tutorial <_static/basic-tutorial.html>`_


API Reference
=============

.. toctree::
   :maxdepth: 2

   DVIDNodeService
   DVIDConnection
   DVIDServerService
   
