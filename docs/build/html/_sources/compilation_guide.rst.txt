
Compilation Guide
=================

Compilation of the package is quite simple (says an expert ...)


Compilation on Windows
----------------------

Windows only supports compilation for python3.5 and higher. This is for the reason that python2 needs an ancient MSCV compiler, which does not support the new coding standard C++14 (also the old compiler created 3x slower code).

For compilation of python version 3.5 or 3.6 on Windows download `Visual Studio 15`_ (Visual Studio 17 also worked for me). Then clone the repository into a folder:

.. _Visual Studio 15: https://www.visualstudio.com/de/vs/older-downloads/

.. code-block:: bash

    git clone --depth 1 https://github.com/qd-cae/qd-cae-python.git
    cd qd-cae-python

The argument ``--depth`` only makes a shallow copy, which saves a lot time and memory. Thereafter run the setup script, the code will automatically compile and install. 

.. code-block:: bash

    python setup.py install


Compilation on Linux
--------------------

Note that in contrast to windows, on Linux the library can be compiled for both, python2 and python3. Check first whether your compiler is new enough.

.. code-block:: bash

    g++ --version

You will need **g++ version 5.4 or higher**. If your compiler is fine, just install as follows:

.. code-block:: bash

    git clone https://github.com/qd-cae/qd-cae-python.git
    cd qd-cae-python
    sudo python setup.py install
    
As a little hint, if you want force python to take your modern compiler version (it may happen that you have multiple g++), simply overwrite the `CC` attribute before the running the setup script:

.. code-block:: bash

    sudo CC=g++-5.4 python setup.py install

Tests
-----

It is highly recommended to run the test suite after compilation to check the installation!

.. code-block:: bash

    python setup.py test

    
Hints on OpenMP
---------------
By default the library uses and supports OpenMP parallelization. Sometimes compatability issues arise though if the path to OpenMP (on your Linux) is not configured correctly (it can confuse the Anaconda OpenMP with your system one). To check if OpenMP is really an issue during compilation, you can optionally disable OpenMP in the top of the ``setup.py`` with ``use_openmp=False`` and the library will be compiled and run without it.
    
    
Compilation with FEMZIP support
-------------------------------

Important note: The installation skips femzip if the libraries are missing (checks if folder with link libraries exists). If it can not find the folder, then the compiler definition :code:`QD_USE_FEMZIP` is not used. This cancels out all code in the source files regarding FEMZIP.

If one wants to compile with FEMZIP support, download the libraries from the `Sidact Homepage`_. There are two things to download, the actual femunzip library:

 - Windows: **FEMZIP_[version]_dyna_NO_OMP_Windows_VS2012_MD_x64.tar.gz**
 - Linux: **FEMZIP_[version]_dyna_NO_OMP_Linux_64Bit.tar.gz**

as well as the link libraries (**FEMUNZIPLIB-LPR_[version]_link_libraries.tar.gz**). Put all content into a folder and register the path in the header of the `setup.py`. 

.. code-block:: python

    femzip_path_windows = "path/to/femzip/windows/libs"
    femzip_path_linux   = "path/to/femzip/linux/libs"



.. _Sidact Homepage: http://www.sidact.com/

