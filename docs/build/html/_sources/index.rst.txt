.. qd documentation master file, created by
   sphinx-quickstart on Thu Mar 30 20:23:40 2017.

Welcome to qd's documentation!
==============================

qd is python3 library for Computer Aided Engineering (CAE). I've developed this library, since I was very unhappy with many expensive commercial software tools in our field of work. I believe many new scientific ideas emerge from freedom and simplicity, therefore the main focus of this project was on a good API and Speed. As a result memory took quite some, so watch you RAM when you load a full vehicle.

.. warning::
    This library is maintained for bugfixes only. Migrate to [lasso-python](https://github.com/lasso-gmbh/lasso-python) if possible. Note please, that not all features are available in lasso-python.

Modules
=======

.. toctree::
   :maxdepth: 3

   qd_cae_dyna
   qd_cae_beta
   qd_numerics

Compilation Guide
=================

Try installation of the precompiled python wheels in the `dist folder`_ before compiling it yourself. If your python distribution doesn't match any of the wheels, then you're in the right spot here.

.. _dist folder: https://github.com/qd-cae/qd-cae-python/tree/master/dist

.. toctree::
    :maxdepth: 2

    compilation_guide

Developer Guide
===============

This section is for anyone, who wants to extend this library or build a new tool from it.

.. toctree::
    :maxdepth: 2

    developer_guide
    
    
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
